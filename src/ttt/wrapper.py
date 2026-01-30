import copy
import math

import torch


def save_embeddings(model):
    """Save input_patch_embedding state and return (state_dict copy, layer ref).

    Args:
        model: Chronos2Model instance (pipeline.model)

    Returns:
        Tuple of (deep-copied state_dict, embedding layer reference)
    """
    emb_layer = model.input_patch_embedding
    return copy.deepcopy(emb_layer.state_dict()), emb_layer


def restore_embeddings(emb_layer, original_state):
    """Restore embedding layer weights from saved state.

    Args:
        emb_layer: the input_patch_embedding layer
        original_state: state_dict to restore
    """
    emb_layer.load_state_dict(original_state)


def ttt_step(model, context, n_mask, optimizer):
    """One TTT optimization step.

    Splits context into practice (input) and target (last n_mask values).
    Uses the model's built-in forward with future_target, which computes
    quantile loss in the normalized space — matching how the model was
    pretrained.

    Args:
        model: Chronos2Model instance (pipeline.model)
        context: (batch_size, context_length) tensor
        n_mask: number of values to mask from the end
        optimizer: optimizer for embedding layer parameters

    Returns:
        loss value (float)
    """
    # Split into practice context and target
    practice_context = context[:, :-n_mask]
    target = context[:, -n_mask:]

    # Number of output patches needed to cover n_mask values
    output_patch_size = model.chronos_config.output_patch_size
    num_output_patches = math.ceil(n_mask / output_patch_size)

    optimizer.zero_grad()

    # Forward pass with future_target — the model normalizes the target
    # with the same loc_scale as the context and computes quantile loss
    # in that normalized space, before instance_norm.inverse().
    output = model(
        context=practice_context,
        future_target=target,
        num_output_patches=num_output_patches,
    )

    loss = output.loss
    loss.backward()
    optimizer.step()

    return loss.item()


class TTTChronos:
    """Chronos-2 wrapper with Test-Time Training adaptation.

    Before making a prediction, adapts the input_patch_embedding layer
    by running a self-supervised "practice exam" on the recent context:
    mask the last n_mask known values, predict them, and update embeddings
    to minimize prediction error. Then predict the actual future with
    the adapted embeddings, and reset afterwards.

    Usage:
        model = TTTChronos(pipeline, n_mask=3, ttt_steps=5, lr=1e-3)
        forecast = model.predict(context, prediction_length=96)
    """

    def __init__(self, pipeline, n_mask=3, ttt_steps=5, lr=1e-3):
        self.pipeline = pipeline
        self.n_mask = n_mask
        self.ttt_steps = ttt_steps
        self.lr = lr

    def predict(self, context, prediction_length):
        """Predict with TTT adaptation on the input_patch_embedding.

        1. Saves embedding weights
        2. Freezes all params except input_patch_embedding
        3. Runs TTT adaptation loop (prints loss each step)
        4. Predicts with adapted embeddings via pipeline.predict()
        5. Resets embeddings to original state

        Args:
            context: input time series, shape (context_length,) or
                     (batch_size, context_length)
            prediction_length: number of future steps to forecast

        Returns:
            forecast from pipeline.predict() - list of tensors,
            each of shape (n_variates, n_quantiles, prediction_length)
        """
        model = self.pipeline.model

        # Save original context for pipeline.predict() (handles its own device/format)
        original_context = context

        # Ensure 2D (batch, length) for TTT adaptation
        if context.ndim == 1:
            context = context.unsqueeze(0)
        context = context.to(device=model.device, dtype=torch.float32)

        # 1. Save original embeddings
        original_state, emb_layer = save_embeddings(model)

        # 2. Freeze all params, enable only embedding layer
        for param in model.parameters():
            param.requires_grad_(False)
        emb_layer.requires_grad_(True)

        optimizer = torch.optim.Adam(emb_layer.parameters(), lr=self.lr)

        # 3. TTT adaptation loop
        model.train()
        for step in range(self.ttt_steps):
            loss = ttt_step(model, context, self.n_mask, optimizer)
            print(f"  TTT step {step}: loss={loss:.4f}")

        # 4. Inference with adapted embeddings
        model.eval()

        # Convert context to format pipeline.predict() expects
        if original_context.ndim == 1:
            pipeline_input = [original_context]
        elif original_context.ndim == 2:
            pipeline_input = [original_context[i] for i in range(original_context.shape[0])]
        else:
            pipeline_input = original_context

        forecast = self.pipeline.predict(pipeline_input, prediction_length=prediction_length)

        # 5. Reset embeddings and requires_grad state
        restore_embeddings(emb_layer, original_state)
        for param in model.parameters():
            param.requires_grad_(True)

        return forecast
