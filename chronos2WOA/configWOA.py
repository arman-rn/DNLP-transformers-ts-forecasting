# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Authors: Abdul Fatir Ansari <ansarnd@amazon.com>

from dataclasses import dataclass, field
from typing import List, Literal

from transformers.configuration_utils import PretrainedConfig


class Chronos2CoreConfig(PretrainedConfig):
    """
    HF transformers-style pretrained model config for Chronos-2.0, based on T5Config.

    Arguments
    ----------
    d_model
        Size model's hidden states, by default 512
    d_kv
        Size of the key, query, value projections per attention head, by default 64
    d_ff
        Size of the intermediate feed forward layers, by default 2048
    num_layers
        Number of hidden layers in the encoder, by default 6
    num_heads
        Number of attention heads for each attention layer, by default 8
    dropout_rate
        The ratio for all dropout layers, by default 0.1
    layer_norm_epsilon
        The epsilon used by the layer normalization layers, by default 1e-6
    initializer_factor
        A factor for initializing all weight matrices, by default 0.05
    feed_forward_proj
        Type of feed forward layer to be used, by default "relu"
    vocab_size
        Size of vocabulary for special tokens, by default 2
    pad_token_id
        Token ID for padding/missing value token, by default 0
    rope_theta
        The base theta for rotary position embedding (RoPE), by default 10000.0
    attn_implementation
        The attention implementation to use. Options: "eager" or "sdpa", by default None (uses "sdpa")
    """

    model_type = "t5"
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
        "head_dim": "d_kv",
    }

    def __init__(
        self,
        d_model: int = 512,
        d_kv: int = 64,
        d_ff: int = 2048,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        layer_norm_epsilon: float = 1e-6,
        initializer_factor: float = 0.05,
        feed_forward_proj: str = "relu",
        vocab_size: int = 2,
        pad_token_id: int = 0,
        rope_theta: float = 10000.0,
        attn_implementation: Literal["eager", "sdpa"] | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.rope_theta = rope_theta

        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"

        assert not self.is_gated_act, "gated activation is not supported"

        # Attention implementation - default to "sdpa" if not specified
        attn_implementation = attn_implementation or "sdpa"
        assert attn_implementation in ["eager", "sdpa"], f"attn_implementation {attn_implementation} not supported"

        # unused
        kwargs.pop("is_encoder_decoder", None)
        kwargs.pop("eos_token_id", None)

        super().__init__(
            pad_token_id=pad_token_id, is_encoder_decoder=False, attn_implementation=attn_implementation, **kwargs
        )



@dataclass
class Chronos2ForecastingConfig:
    # --- 1. Standard Fields (Give ALL of them defaults to avoid TypeError) ---
    context_length: int = 512
    prediction_length: int = 64
    input_patch_size: int = 64
    output_patch_size: int = 64
    num_samples: int = 20
    
    # These are usually lists or optional, so we use field(default_factory=...) if they are mutable
    # But for simple types, None or values work:
    
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    use_reg_token: bool = True
    use_arcsinh: bool = False
    max_output_patches: int = 1
    time_encoding_scale: int | None = None
    
    # --- 2. WOA Adaptive Fields ---
    input_patch_stride: int = 64
    sensitivity: float = 25.0
    min_stride: int = 1
    uniform_stride: int | None = None  # Ablation: if set, bypass volatility and use this stride for every patch
    per_sequence_volatility: bool = False  # If True, compute volatility per-sequence and pad to max length per batch
    use_rezero_stride: bool = False  # If True, gate stride_embedding with a learnable scalar alpha (init 0)
    
    @classmethod
    def editable_fields(cls) -> list[str]:
        return ["context_length", "max_output_patches"]