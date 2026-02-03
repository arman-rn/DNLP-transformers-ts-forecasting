# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Authors: Abdul Fatir Ansari <ansarnd@amazon.com>

import copy
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
from einops import rearrange, repeat
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from chronos.chronos_bolt import InstanceNorm, Patch

from configWOA import Chronos2CoreConfig, Chronos2ForecastingConfig
from layers import (
    MHA,
    MLP,
    AttentionOutput,
    Chronos2LayerNorm,
    FeedForward,
    GroupSelfAttention,
    ResidualBlock,
    TimeSelfAttention,
)


@dataclass
class Chronos2EncoderBlockOutput(ModelOutput):
    hidden_states: torch.Tensor | None = None
    time_self_attn_weights: torch.Tensor | None = None
    group_self_attn_weights: torch.Tensor | None = None


class Chronos2EncoderBlock(nn.Module):
    def __init__(self, config: Chronos2CoreConfig):
        super().__init__()
        assert not config.is_decoder

        self.layer = nn.ModuleList()
        self.layer.append(TimeSelfAttention(config))
        self.layer.append(GroupSelfAttention(config))
        self.layer.append(FeedForward(config))

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        group_time_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> Chronos2EncoderBlockOutput:
        # apply time attention
        time_self_attn_outputs: AttentionOutput = self.layer[0](
            hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = time_self_attn_outputs[0]

        # apply group attention
        group_self_attn_outputs: AttentionOutput = self.layer[1](
            hidden_states, attention_mask=group_time_mask, output_attentions=output_attentions
        )
        hidden_states = group_self_attn_outputs[0]

        # apply feed forward layer
        hidden_states = self.layer[2](hidden_states)

        return Chronos2EncoderBlockOutput(
            hidden_states=hidden_states,
            time_self_attn_weights=time_self_attn_outputs.attn_weights,
            group_self_attn_weights=group_self_attn_outputs.attn_weights,
        )


@dataclass
class Chronos2EncoderOutput(ModelOutput):
    last_hidden_state: torch.Tensor | None = None
    all_time_self_attn_weights: tuple[torch.Tensor, ...] | None = None
    all_group_self_attn_weights: tuple[torch.Tensor, ...] | None = None


class Chronos2Encoder(nn.Module):
    def __init__(self, config: Chronos2CoreConfig):
        super().__init__()
        assert not config.is_decoder

        self.block = nn.ModuleList([Chronos2EncoderBlock(config) for i in range(config.num_layers)])
        self.final_layer_norm = Chronos2LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    @staticmethod
    def _expand_and_invert_time_attention_mask(
        attention_mask: torch.Tensor, floating_type: torch.dtype
    ) -> torch.Tensor:
        assert attention_mask.ndim == 2, "attention_mask must have shape (batch, seq_len)"

        # Add new dims for attention heads and q_len
        attention_mask = attention_mask[:, None, None, :]

        # Invert binary mask to float mask which can be added to attention scores
        attention_mask = attention_mask.to(dtype=floating_type)
        attention_mask = (1.0 - attention_mask) * torch.finfo(floating_type).min
        return attention_mask

    @staticmethod
    def _construct_and_invert_group_time_mask(
        group_ids: torch.Tensor, attention_mask: torch.Tensor, floating_type: torch.dtype
    ) -> torch.Tensor:
        # construct group_mask (batch, batch) from group ids
        # a cell is True if both row and col had the same group id
        group_mask = group_ids[:, None] == group_ids[None, :]
        # outer product of group_mask and attention_mask (time_mask)
        # group_time_mask combines group and time masks to ensure that attention only uses
        # tokens from the same group which are also not masked in time
        group_time_mask = torch.einsum("qb, bt -> qbt", group_mask, attention_mask)

        if torch.is_floating_point(group_time_mask):
            # this ensures that mixed precision training does not overflow
            floating_type = group_time_mask.dtype

        # reshape mask to shape of attention scores
        group_time_mask = rearrange(group_time_mask, "q b t -> t 1 q b")
        group_time_mask = (1.0 - group_time_mask) * torch.finfo(floating_type).min

        return group_time_mask

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        *,
        group_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> Chronos2EncoderOutput:
        batch_size, seq_length = inputs_embeds.size()[:-1]

        if position_ids is None:
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=inputs_embeds.device).unsqueeze(0)

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, device=inputs_embeds.device, dtype=inputs_embeds.dtype)

        # make the time attention mask broadcastable to attention scores (batch, n_heads, q_len, kv_len) and invert
        extended_attention_mask = self._expand_and_invert_time_attention_mask(attention_mask, inputs_embeds.dtype)

        # construct group time mask
        group_time_mask = self._construct_and_invert_group_time_mask(group_ids, attention_mask, inputs_embeds.dtype)

        all_time_self_attentions: tuple[torch.Tensor, ...] = ()
        all_group_self_attentions: tuple[torch.Tensor, ...] = ()

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module) in enumerate(self.block):
            layer_outputs: Chronos2EncoderBlockOutput = layer_module(
                hidden_states,
                position_ids=position_ids,
                attention_mask=extended_attention_mask,
                group_time_mask=group_time_mask,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                assert layer_outputs.time_self_attn_weights is not None
                assert layer_outputs.group_self_attn_weights is not None

                all_time_self_attentions = (*all_time_self_attentions, layer_outputs.time_self_attn_weights)
                all_group_self_attentions = (*all_group_self_attentions, layer_outputs.group_self_attn_weights)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return Chronos2EncoderOutput(
            last_hidden_state=hidden_states,
            all_time_self_attn_weights=all_time_self_attentions,
            all_group_self_attn_weights=all_group_self_attentions,
        )


@dataclass
class Chronos2Output(ModelOutput):
    loss: torch.Tensor | None = None
    quantile_preds: torch.Tensor | None = None
    enc_time_self_attn_weights: tuple[torch.Tensor, ...] | None = None
    enc_group_self_attn_weights: tuple[torch.Tensor, ...] | None = None


class Chronos2Model(PreTrainedModel):
    config_class = Chronos2CoreConfig  # type: ignore[assignment]
    _supports_long_horizon: bool = True
    _supports_future_covariates: bool = True
    _supports_sdpa: bool = True

    def __init__(self, config: Chronos2CoreConfig):
        assert hasattr(config, "chronos_config"), "Not a valid Chronos config"

        super().__init__(config)
        self.config: Chronos2CoreConfig
        self.model_dim = config.d_model

        config.chronos_config["time_encoding_scale"] = config.chronos_config.get(
            "time_encoding_scale", config.chronos_config["context_length"]
        )
        self.chronos_config = Chronos2ForecastingConfig(**config.chronos_config)

        assert self.chronos_config.input_patch_size == self.chronos_config.output_patch_size, (
            "input_patch_size and output_patch_size sizes must be equal, "
            f"but found {self.chronos_config.input_patch_size} and {self.chronos_config.output_patch_size}"
        )

        # Only [PAD] token (and [REG] token)
        if self.chronos_config.use_reg_token:
            config.reg_token_id = 1

        config.vocab_size = 2 if self.chronos_config.use_reg_token else 1
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        
        
        # --- NEW: STRIDE EMBEDDING LAYER --- for the time alignment 
        # We create an embedding for every possible stride size (from 1 up to max stride (input patch stride))
        self.stride_embedding = nn.Embedding(
            num_embeddings=self.chronos_config.input_patch_stride + 1, 
            embedding_dim=config.d_model
        )
        
        # Initialize with small weights so it doesn't disrupt the model initially
        self.stride_embedding.weight.data.normal_(mean=0.0, std=0.01)
        # -----------------------------------
        
        # Input patch embedding layer
        self.input_patch_embedding = ResidualBlock(
            # x3 for [time_embedding, patch, patch_mask]
            in_dim=self.chronos_config.input_patch_size * 3,
            h_dim=config.d_ff,
            out_dim=config.d_model,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,
        )

        # patching layer
        self.patch = Patch(
            patch_size=self.chronos_config.input_patch_size, patch_stride=self.chronos_config.input_patch_stride
        )

        # instance normalization, also referred to as "scaling" in Chronos and GluonTS
        self.instance_norm = InstanceNorm(use_arcsinh=self.chronos_config.use_arcsinh)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        self.encoder = Chronos2Encoder(encoder_config)

        self.num_quantiles = len(self.chronos_config.quantiles)
        quantiles = torch.tensor(self.chronos_config.quantiles, dtype=self.dtype)
        self.quantiles: torch.Tensor
        self.register_buffer("quantiles", quantiles, persistent=False)

        self.output_patch_embedding = ResidualBlock(
            in_dim=config.d_model,
            h_dim=config.d_ff,
            out_dim=self.num_quantiles * self.chronos_config.output_patch_size,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module):
        super()._init_weights(module)
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, Chronos2LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, MLP):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, MHA):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            kv_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * kv_proj_dim) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * kv_proj_dim) ** -0.5))
        elif isinstance(module, (Chronos2Model)):
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
            
        elif isinstance(module, nn.Embedding) and hasattr(self, 'stride_embedding') and module is self.stride_embedding:
            module.weight.data.normal_(mean=0.0, std=factor * 1.0)
        
        elif isinstance(module, ResidualBlock):
            module.hidden_layer.weight.data.normal_(
                mean=0.0,
                std=factor * (module.hidden_layer.weight.size(-1) ** -0.5),
            )
            if hasattr(module.hidden_layer, "bias") and module.hidden_layer.bias is not None:
                module.hidden_layer.bias.data.zero_()

            module.residual_layer.weight.data.normal_(
                mean=0.0,
                std=factor * (module.residual_layer.weight.size(-1) ** -0.5),
            )
            if hasattr(module.residual_layer, "bias") and module.residual_layer.bias is not None:
                module.residual_layer.bias.data.zero_()

            module.output_layer.weight.data.normal_(
                mean=0.0, std=factor * (module.output_layer.weight.size(-1) ** -0.5)
            )
            if hasattr(module.output_layer, "bias") and module.output_layer.bias is not None:
                module.output_layer.bias.data.zero_()
        
        
        
        
    def _validate_input(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
        group_ids: torch.Tensor | None,
        future_covariates: torch.Tensor | None,
        future_covariates_mask: torch.Tensor | None,
        num_output_patches: int,
        future_target: torch.Tensor | None,
        future_target_mask: torch.Tensor | None,
    ):
        output_patch_size = self.chronos_config.output_patch_size
        if context.ndim != 2:
            raise ValueError(f"context must have shape (batch_size, context_length), found: {tuple(context.shape)}")
        if context_mask is not None and context_mask.shape != context.shape:
            raise ValueError(f"mask must have shape {tuple(context.shape)}, found: {tuple(context_mask.shape)}")
        if future_covariates is not None:
            if future_covariates.shape[0] != context.shape[0] or future_covariates.ndim != 2:
                raise ValueError(
                    f"future_covariates must have shape (batch_size={context.shape[0]}, future_length), found: {tuple(future_covariates.shape)}"
                )
            if future_covariates.shape[-1] > num_output_patches * output_patch_size:
                raise ValueError(
                    f"{num_output_patches=} must be large enough to accommodate the length of future_covariates, "
                    f"found: {future_covariates.shape[-1]} > {num_output_patches} * {output_patch_size}"
                )
            if future_target is not None and future_target.shape != future_covariates.shape:
                raise ValueError(
                    f"future_target must have the same shape as future_covariates, found: {tuple(future_target.shape)} and {tuple(future_covariates.shape)}"
                )
        if future_covariates_mask is not None:
            if future_covariates is None:
                raise ValueError("future_covariates must be provided if future_covariates_mask is provided")
            if future_covariates_mask.shape != future_covariates.shape:
                raise ValueError(
                    f"future_covariates_mask must have the same shape as future_covariates, "
                    f"found: {tuple(future_covariates_mask.shape)} and {tuple(future_covariates.shape)}"
                )
        if group_ids is not None and group_ids.shape != (context.shape[0],):
            raise ValueError(f"group_ids must have shape (batch_size,), found: {tuple(group_ids.shape)}")
        if future_target is not None:
            if future_target.shape[0] != context.shape[0] or future_target.ndim != 2:
                raise ValueError(
                    f"future_target must have shape (batch_size={context.shape[0]}, future_length), found: {tuple(future_target.shape)}"
                )
            if future_target.shape[-1] > output_patch_size * num_output_patches:
                raise ValueError(
                    f"{num_output_patches=} must be large enough to accommodate the length of future_target, "
                    f"found: {future_target.shape[-1]} > {num_output_patches} * {output_patch_size}"
                )
        if future_target_mask is not None:
            if future_target is None:
                raise ValueError("future_target must be provided if future_target_mask is provided")
            if future_target_mask.shape != future_target.shape:
                raise ValueError(
                    f"future_target_mask must have the same shape as future_target, found: {tuple(future_target_mask.shape)} and {tuple(future_target.shape)}"
                )
    
    
    
    
    
    
    def _prepare_patched_future(
        self,
        future_covariates: torch.Tensor | None,
        future_covariates_mask: torch.Tensor | None,
        loc_scale: tuple[torch.Tensor, torch.Tensor],
        num_output_patches: int,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        from einops import repeat, rearrange

        output_patch_size = self.chronos_config.output_patch_size
        
        # --- FIX 1: Loc/Scale Broadcasting ---
        loc, scale = loc_scale
        if loc.shape[1] > 1:
            loc = loc[:, 0:1, :]
            scale = scale[:, 0:1, :]
        loc_scale = (loc, scale) 

        if future_covariates is not None:
            # --- FIX 2: Handle 2D Inputs ---
            if future_covariates.ndim == 2:
                future_covariates = future_covariates.unsqueeze(-1)
            
            future_covariates, _ = self.instance_norm(future_covariates, loc_scale)
            future_covariates = future_covariates.to(self.dtype)

            if future_covariates_mask is None:
                future_covariates_mask = torch.isnan(future_covariates).logical_not().to(future_covariates.dtype)
            
            if future_covariates_mask.ndim == 2:
                future_covariates_mask = future_covariates_mask.unsqueeze(-1)

            future_covariates = torch.where(future_covariates_mask > 0.0, future_covariates, 0.0)
            
            # --- RESTORED SAFETY CHECK (Original Logic) ---
            if torch.isnan(future_covariates).any():
                 # We can just warn or zero fill, but raising error is safer
                 # For training stability, let's zero fill instead of crashing
                 future_covariates = torch.nan_to_num(future_covariates, nan=0.0)

            # Padding logic
            curr_len = future_covariates.shape[1]
            needed_len = num_output_patches * output_patch_size
            
            if needed_len > curr_len:
                padding = needed_len - curr_len
                pad_shape = (batch_size, padding, future_covariates.shape[-1])
                future_covariates = torch.cat(
                    [future_covariates, torch.zeros(pad_shape, device=self.device, dtype=self.dtype)], dim=1
                )
                future_covariates_mask = torch.cat(
                    [future_covariates_mask, torch.zeros(pad_shape, device=self.device, dtype=self.dtype)], dim=1
                )

            # Patching (Keep 3D structure [B, N, P, 1])
            patched_future_covariates = rearrange(
                future_covariates, "b (n p) f -> b n p f", n=num_output_patches, p=output_patch_size
            )
            patched_future_covariates_mask = rearrange(
                future_covariates_mask, "b (n p) f -> b n p f", n=num_output_patches, p=output_patch_size
            )
        else:
            patched_future_covariates = torch.zeros(
                batch_size, num_output_patches, output_patch_size, 1, device=self.device, dtype=self.dtype
            )
            patched_future_covariates_mask = torch.zeros(
                batch_size, num_output_patches, output_patch_size, 1, device=self.device, dtype=self.dtype
            )

        # Time Encoding
        final_future_length = num_output_patches * output_patch_size
        scale_val = self.chronos_config.time_encoding_scale if self.chronos_config.time_encoding_scale else self.chronos_config.context_length

        future_time_enc = torch.arange(start=0, end=int(final_future_length), device=self.device, dtype=torch.float32)
        future_time_enc = (
            repeat(
                future_time_enc,
                "(n p) -> b n p 1", 
                b=batch_size,
                n=num_output_patches,
                p=output_patch_size,
            )
            .div(scale_val)
            .to(self.dtype)
        )

        # --- FIX 3: RESTORE DATA ORDER (Critical) ---
        # 1. Concatenate features at the end: [Batch, N, Patch, 3]
        # (Contains: Time, Value, Mask)
        patched_future = torch.cat(
            [future_time_enc, patched_future_covariates, patched_future_covariates_mask], dim=-1
        )
        
        # 2. PERMUTE to group by feature type: [Batch, N, 3, Patch]
        # This groups all Times together, all Values together, etc.
        patched_future = patched_future.permute(0, 1, 3, 2)
        
        # 3. FLATTEN: [Batch, N, 3 * Patch]
        # Result: [Time0..Time15, Val0..Val15, Mask0..Mask15] -> MATCHES ORIGINAL!
        patched_future = patched_future.reshape(batch_size, num_output_patches, -1)
        
        # Flatten mask for consistency if used elsewhere
        patched_future_covariates_mask = patched_future_covariates_mask.reshape(batch_size, num_output_patches, -1)

        return patched_future, patched_future_covariates_mask
    
    # Input (context): A single long 1D array (e.g., 512 numbers) = [P1, P2, P3, ... P512]
    # Output (patched_context): A 2D matrix (a stack of shorter arrays).

            #Row 1: [[P1 ... P64] (Patch 1)
            #Row 2: [P65 ... P128],...] (Patch 2)
    

    def _prepare_patched_context(
        self, 
        context: torch.Tensor, 
        context_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    
        from einops import repeat 

        # 1. ROBUST SCALING & NORMALIZATION [cite: 198]
        if context_mask is None:
            context_mask = torch.isnan(context).logical_not()
        
        if context.shape[1] > self.chronos_config.context_length:
            context = context[:, -self.chronos_config.context_length :]
            context_mask = context_mask[:, -self.chronos_config.context_length :]

        # Apply standardization and sinh transformation to stabilize variance [cite: 198, 199]
        context, loc_scale = self.instance_norm(context)
    
        if context.ndim == 2: context = context.unsqueeze(-1)
        if context_mask.ndim == 2: context_mask = context_mask.unsqueeze(-1)
        
        context = context.to(self.dtype)
        context_mask = context_mask.to(self.dtype)
    
        loc, scale = loc_scale
        if loc.ndim == 2:
            loc = loc.unsqueeze(-1)
            scale = scale.unsqueeze(-1)
        loc_scale = (loc, scale)

        # 2. ADAPTIVE SELECTION (WOA)
        batch_size, context_length, _ = context.shape
        patch_size = self.chronos_config.input_patch_size
        default_stride = self.chronos_config.input_patch_stride
        sensitivity = self.chronos_config.sensitivity
        min_s = self.chronos_config.min_stride
    
        # Calculate baseline volatility for adaptive braking
        target_seq = context[:, :, 0] 
        all_patches_rigid = target_seq.unfold(1, patch_size, 1)
        all_stds = all_patches_rigid.std(dim=-1)
        baseline_vol = torch.quantile(all_stds, 0.2).item()

        patches_list, masks_list, strides_list, absolute_offsets = [], [], [], []
        cursor = 0

        while cursor + patch_size <= context_length:
            curr_patch = context[:, cursor : cursor + patch_size, :]
            curr_mask = context_mask[:, cursor : cursor + patch_size, :]

            patches_list.append(curr_patch) 
            masks_list.append(curr_mask)
            absolute_offsets.append(cursor) # Real start time index

            # Adaptive stride: smaller step in volatile regions (high density selection)
            curr_target_patch = curr_patch[:, :, 0]
            local_vol = curr_target_patch.std(dim=-1).max().item()
            excess = max(0, local_vol - baseline_vol)
            braking_factor = 1 + (excess * sensitivity) 
            step = int(max(min_s, round(default_stride / braking_factor)))

            strides_list.append(step)
            cursor += step

        # 3. NON-DISTORTED TIME ENCODING [cite: 229]
        # We use the TRUE absolute offsets to anchor patches to the timeline
        max_supported_len = self.chronos_config.time_encoding_scale or self.chronos_config.context_length
        time_enc_patches = []

        for offset in absolute_offsets:
            # Patch time is derived from its actual position in the original series
            # j = [-(T)/C, ..., 0] [cite: 229]
            patch_time = torch.arange(offset, offset + patch_size, device=context.device)
            patch_time = (patch_time - context_length) / max_supported_len
            time_enc_patches.append(patch_time)

        # Create [Batch, NumPatches, PatchSize, 1] tensor
        context_time_enc = torch.stack(time_enc_patches, dim=0).unsqueeze(0).unsqueeze(-1)
        context_time_enc = context_time_enc.expand(batch_size, -1, -1, -1).to(self.dtype)

        # 4. FINAL ASSEMBLY 
        patched_context = torch.stack(patches_list, dim=1) 
        patched_mask = torch.stack(masks_list, dim=1)
        patched_mask = torch.nan_to_num(patched_mask, nan=0.0)
        patched_context = torch.where(patched_mask > 0.0, patched_context, 0.0)

        # Concatenate features: [Time, Value, Mask] -> mapped via residual network 
        final_output = torch.cat([context_time_enc, patched_context, patched_mask], dim=-1)

        # Permute to group by feature type before flattening: [B, N, 3, P] -> [B, N, 3*P]
        num_patches = len(patches_list)
        final_output = final_output.permute(0, 1, 3, 2).reshape(batch_size, num_patches, -1)

        # Binary mask indicating if a patch is valid [cite: 231]
        attention_mask = patched_mask.sum(dim=(-2, -1)) > 0 

        current_strides = torch.tensor(strides_list, device=context.device, dtype=torch.long)
        self._last_strides = current_strides # Store for visualization script

        return final_output, attention_mask, loc_scale, current_strides
    
    def _compute_loss(
        self,
        quantile_preds: torch.Tensor,
        future_target: torch.Tensor,
        future_target_mask: torch.Tensor | None,
        patched_future_covariates_mask: torch.Tensor,
        loc_scale: tuple[torch.Tensor, torch.Tensor],
        num_output_patches: int,
    ) -> torch.Tensor:
        
        from einops import rearrange

        # --- FIX 1: Safety First (Normalization) ---
        # 1. Slice stats to scalars [Batch, 1, 1] to prevent 2048 vs 64 crash
        loc, scale = loc_scale
        if loc.shape[1] > 1:
            loc = loc[:, 0:1, :]
            scale = scale[:, 0:1, :]

        # 2. Handle Univariate Inputs (Safe unsqueeze)
        if future_target.ndim == 2:
            future_target = future_target.unsqueeze(-1) #[B, T, 1]
        
        if future_target_mask is None:
            future_target_mask = torch.isnan(future_target).logical_not()
        if future_target_mask.ndim == 2:
            future_target_mask = future_target_mask.unsqueeze(-1) #[B, T, 1]

        # 3. Manual Normalization (Safe)
        future_target = (future_target.to(self.device) - loc) / scale
        
        # -----------------------------------------------------------
        # ORIGINAL LOGIC RESTORED (Vectorized & Covariate Aware)
        # -----------------------------------------------------------

        # 1. Align Target for Broadcasting: [B, T, 1] -> [B, 1, T]
        # This matches quantile_preds [B, Q, T]
        future_target = rearrange(future_target, "b t f -> b f t") 
        future_target_mask = rearrange(future_target_mask.to(self.device), "b t f -> b f t")
        
        # 2. Pad Target (Original Safety Check)
        # If predictions are longer than target, pad target with zeros
        pred_len = quantile_preds.shape[-1]
        tgt_len = future_target.shape[-1]
        
        if pred_len > tgt_len:
            pad_len = pred_len - tgt_len
            future_target = torch.nn.functional.pad(future_target, (0, pad_len))
            future_target_mask = torch.nn.functional.pad(future_target_mask, (0, pad_len))
        
        quantiles_tensor = torch.tensor(
            self.chronos_config.quantiles, 
            device=self.device, 
            dtype=self.dtype
        )
        
        # 3. Vectorized Quantile Loss (Original Formula)
        # Calculate '2 * abs(...)' exactly like the original
        quantiles = rearrange(quantiles_tensor, "q -> 1 q 1").to(self.device)
        
        # future_target is [B, 1, T], quantile_preds is [B, Q, T] -> Broadcasts to [B, Q, T]
        quantile_loss = 2 * torch.abs(
            (future_target - quantile_preds) * ((future_target <= quantile_preds).float() - quantiles)
        )

        # 4. COVARIATE MASKING (The logic you asked for)
        # We need to reshape the patched mask to match the target timeline [B, 1, T]
        # patched_future_covariates_mask comes in as [Batch, NumPatches, FlattenedDim]
        
        # Reshape: [B, N, P*F] -> [B, 1, T]
        # We assume Univariate F=1, so P*F = P (16)
        inv_future_covariate_mask = rearrange(
            patched_future_covariates_mask, 
            "b n p -> b 1 (n p)"
        )
        
        # Invert it: If it's a known covariate (1), we mask it out (0) for loss.
        # "Don't compute loss on things we already know."
        inv_future_covariate_mask = 1.0 - inv_future_covariate_mask
        
        # 5. Final Mask Composition
        # Loss is valid IF: (Target exists) AND (It is NOT a known covariate)
        loss_mask = future_target_mask.float() * inv_future_covariate_mask
        
        loss = quantile_loss * loss_mask
        
        # 6. Aggregation (Mean over T, Sum over Q, Mean over B)
        return loss.mean(dim=-1).sum(dim=-1).mean()

    def encode(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        future_covariates_mask: torch.Tensor | None = None,
        num_output_patches: int = 1,
        future_target: torch.Tensor | None = None,
        future_target_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ):
        self._validate_input(
            context=context,
            context_mask=context_mask,
            future_covariates=future_covariates,
            future_covariates_mask=future_covariates_mask,
            group_ids=group_ids,
            num_output_patches=num_output_patches,
            future_target=future_target,
            future_target_mask=future_target_mask,
        )

        batch_size = context.shape[0]
        
        # 1. PREPARE CONTEXT (This calls your Adaptive Logic)
        # It populates self._last_strides internally
        patched_context, attention_mask, loc_scale, current_strides = self._prepare_patched_context(
            context=context, context_mask=context_mask
        )
        num_context_patches = attention_mask.shape[-1]

        # 2. GET BASE EMBEDDINGS
        # shape: (batch, num_context_patches, d_model)
     
        input_embeds = self.input_patch_embedding(patched_context)

        # Now use the local current_strides variable
         # Expand it to match the batch size: [Batch, NumPatches]
        stride_batch = current_strides.unsqueeze(0).expand(batch_size, -1)

        stride_embeds_context = self.stride_embedding(stride_batch)
        input_embeds = input_embeds + stride_embeds_context
        # -----------------------------------------------

        # 3. HANDLE [REG] TOKEN
        if self.chronos_config.use_reg_token:
            reg_input_ids = torch.full((batch_size, 1), self.config.reg_token_id, device=input_embeds.device)
            reg_embeds = self.shared(reg_input_ids)
            
            # NOTE: We do NOT add stride embeddings to the REG token (it has no stride)
            # [ Patch+Stride, Patch+Stride, ..., REG_Pure ]
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat(
                [attention_mask.to(self.dtype), torch.ones_like(reg_input_ids).to(self.dtype)], dim=-1
            )

        # 4. PREPARE FUTURE
        patched_future, patched_future_covariates_mask = self._prepare_patched_future(
            future_covariates=future_covariates,
            future_covariates_mask=future_covariates_mask,
            loc_scale=loc_scale,
            num_output_patches=num_output_patches,
            batch_size=batch_size,
        )
        future_attention_mask = torch.ones(batch_size, num_output_patches, dtype=self.dtype, device=self.device)

        # get future embeddings
        future_embeds: torch.Tensor = self.input_patch_embedding(patched_future)

        # --- NEW: INJECT STRIDE EMBEDDINGS (FUTURE) ---
        # The future always moves at "Standard Stride" (e.g., 64).
        # We must tell the model this so it knows the "time warping" has stopped.
        default_stride = self.chronos_config.input_patch_stride
        
        # Create a tensor filled with the default stride ID
        future_strides = torch.full(
            (batch_size, num_output_patches), 
            default_stride, 
            dtype=torch.long, 
            device=self.device
        )
        
        # Lookup and Add
        stride_embeds_future = self.stride_embedding(future_strides)
        future_embeds = future_embeds + stride_embeds_future
        # ----------------------------------------------

        # 5. CONCATENATE EVERYTHING
        input_embeds = torch.cat([input_embeds, future_embeds], dim=-2)
        attention_mask = torch.cat([attention_mask, future_attention_mask], dim=-1)

        if group_ids is None:
            group_ids = torch.arange(batch_size, dtype=torch.long, device=self.device)

        encoder_outputs: Chronos2EncoderOutput = self.encoder(
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
            group_ids=group_ids,
            output_attentions=output_attentions,
        )
        
        return encoder_outputs, loc_scale, patched_future_covariates_mask, num_context_patches

    def forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        future_covariates_mask: torch.Tensor | None = None,
        num_output_patches: int = 1,
        future_target: torch.Tensor | None = None,
        future_target_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> Chronos2Output:
        
        from einops import rearrange # Ensure import

        batch_size = context.shape[0]
        
        # 1. Encode Context
        encoder_outputs, loc_scale, patched_future_covariates_mask, num_context_patches = self.encode(
            context=context,
            context_mask=context_mask,
            group_ids=group_ids,
            future_covariates=future_covariates,
            future_covariates_mask=future_covariates_mask,
            num_output_patches=num_output_patches,
            future_target=future_target,
            future_target_mask=future_target_mask,
            output_attentions=output_attentions,
        )
        
        # --- FIX: Slice loc_scale to Scalars [Batch, 1, 1] ---
        # This prevents the "2048 vs 4" error in loss calculation and unscaling
        loc, scale = loc_scale
        if loc.shape[1] > 1:
            loc = loc[:, 0:1, :]
            scale = scale[:, 0:1, :]
        loc_scale = (loc, scale) # Re-pack safe scalars
        # -----------------------------------------------------

        hidden_states: torch.Tensor = encoder_outputs[0]
        
        # 2. Generate Predictions
        # slice the last num_output_patches hidden states
        forecast_embeds = hidden_states[:, -num_output_patches:]
        quantile_preds: torch.Tensor = self.output_patch_embedding(forecast_embeds)
        
        quantile_preds = rearrange(
            quantile_preds,
            "b n (q p) -> b q (n p)",
            n=num_output_patches,
            q=self.num_quantiles,
            p=self.chronos_config.output_patch_size,
        )

        # 3. Compute Loss (Now safe because we fixed loc_scale above)
        loss = (
            self._compute_loss(
                quantile_preds=quantile_preds,
                future_target=future_target,
                future_target_mask=future_target_mask,
                patched_future_covariates_mask=patched_future_covariates_mask,
                loc_scale=loc_scale, # Passing the fixed scalar stats
                num_output_patches=num_output_patches,
            )
            if future_target is not None
            else None
        )

        # 4. Unscale Predictions (Now safe because loc_scale is [B, 1, 1])
        # We need to flatten to [Batch, Time] for inverse transform, then reshape back
        # Actually, inverse expects [B, Q, T] or similar. Let's check dims.
        # quantile_preds is [Batch, Quantiles, Time]
        
        # The inverse method usually broadcasts on the last dim.
        # We might need to handle the Quantile dimension manually.
        
        # Move Quantiles to Batch dimension to use standard inverse()
        # [B, Q, T] -> [B*Q, T]
        b, q, t = quantile_preds.shape
        quantile_preds_flat = rearrange(quantile_preds, "b q t -> (b q) t")
        
        # We also need to repeat loc_scale to match B*Q
        loc_repeated = loc.repeat_interleave(q, dim=0)
        scale_repeated = scale.repeat_interleave(q, dim=0)
        
        # Handle 2D/3D mismatch in inverse()
        # If inverse expects [Batch, Time, Features], we need to add feature dim
        quantile_preds_flat = quantile_preds_flat.unsqueeze(-1) # [BQ, T, 1]
        
        # Perform Inverse
        # Note: We use manual calculation to be 100% safe against shape errors
        quantile_preds_flat = quantile_preds_flat * scale_repeated + loc_repeated
        
        # Reshape back: [BQ, T, 1] -> [B, Q, T]
        quantile_preds = rearrange(quantile_preds_flat.squeeze(-1), "(b q) t -> b q t", b=b, q=q)

        return Chronos2Output(
            loss=loss,
            quantile_preds=quantile_preds,
            enc_time_self_attn_weights=encoder_outputs.all_time_self_attn_weights,
            enc_group_self_attn_weights=encoder_outputs.all_group_self_attn_weights,
        )
