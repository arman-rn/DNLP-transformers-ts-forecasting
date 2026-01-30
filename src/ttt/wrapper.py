"""TTT wrapper for Chronos-2.

Supports choosing which layer(s) to adapt via the `target` parameter:
  "input"  — input_patch_embedding
  "output" — output_patch_embedding (default, receives direct gradient from loss)
  "both"   — both patch embeddings

Also supports optimizer choice: "adam" (default) or "sgd".
"""

import copy
import math

import torch


def get_ttt_layers(model, target):
    """Return list of (name, module) pairs for the TTT target layer(s).

    Args:
        model: Chronos2Model instance (pipeline.model)
        target: which layer(s) to adapt:
            "input"  — input_patch_embedding only
            "output" — output_patch_embedding only
            "both"   — both input and output patch embeddings

    Returns:
        List of (name, module) tuples
    """
    if target == "input":
        return [("input_patch_embedding", model.input_patch_embedding)]
    elif target == "output":
        return [("output_patch_embedding", model.output_patch_embedding)]
    elif target == "both":
        return [
            ("input_patch_embedding", model.input_patch_embedding),
            ("output_patch_embedding", model.output_patch_embedding),
        ]
    else:
        raise ValueError(f"Unknown target '{target}', expected 'input', 'output', or 'both'")


def save_layers(layers):
    """Save state dicts for a list of (name, module) pairs.

    Returns:
        List of (module, deep-copied state_dict) tuples
    """
    return [(mod, copy.deepcopy(mod.state_dict())) for _, mod in layers]


def restore_layers(saved):
    """Restore modules from saved state."""
    for mod, state in saved:
        mod.load_state_dict(state)


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
        optimizer: optimizer for target layer parameters

    Returns:
        loss value (float)
    """
    practice_context = context[:, :-n_mask]
    target = context[:, -n_mask:]

    output_patch_size = model.chronos_config.output_patch_size
    num_output_patches = math.ceil(n_mask / output_patch_size)

    optimizer.zero_grad()

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
    """Chronos-2 wrapper with Test-Time Training — configurable target layer.

    Before making a prediction, adapts the chosen layer(s) by running a
    self-supervised "practice exam" on the recent context: mask the last
    n_mask known values, predict them, and update weights to minimize
    prediction error. Then predict the actual future with adapted weights,
    and reset afterwards.

    Usage:
        model = TTTChronos(pipeline, n_mask=16, ttt_steps=5, lr=1e-4, target="output")
        forecast = model.predict(context, prediction_length=96)
    """

    def __init__(self, pipeline, n_mask=16, ttt_steps=5, lr=1e-4, target="output",
                 optimizer="adam"):
        self.pipeline = pipeline
        self.n_mask = n_mask
        self.ttt_steps = ttt_steps
        self.lr = lr
        self.target = target
        self.optimizer_type = optimizer

    def predict(self, context, prediction_length):
        """Predict with TTT adaptation on the chosen target layer(s).

        Args:
            context: input time series, shape (context_length,) or
                     (batch_size, context_length)
            prediction_length: number of future steps to forecast

        Returns:
            forecast from pipeline.predict() - list of tensors,
            each of shape (n_variates, n_quantiles, prediction_length)
        """
        model = self.pipeline.model
        original_context = context

        if context.ndim == 1:
            context = context.unsqueeze(0)
        context = context.to(device=model.device, dtype=torch.float32)

        # 1. Get target layers and save their state
        layers = get_ttt_layers(model, self.target)
        saved = save_layers(layers)

        # 2. Freeze all params, enable only target layer(s)
        for param in model.parameters():
            param.requires_grad_(False)
        for _, mod in layers:
            mod.requires_grad_(True)

        trainable_params = []
        for _, mod in layers:
            trainable_params.extend(mod.parameters())

        if self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(trainable_params, lr=self.lr)
        else:
            optimizer = torch.optim.Adam(trainable_params, lr=self.lr)

        # 3. TTT adaptation loop
        model.train()
        for step in range(self.ttt_steps):
            loss = ttt_step(model, context, self.n_mask, optimizer)
            print(f"  TTT step {step}: loss={loss:.4f}")

        # 4. Inference with adapted weights
        model.eval()

        if original_context.ndim == 1:
            pipeline_input = [original_context]
        elif original_context.ndim == 2:
            pipeline_input = [original_context[i] for i in range(original_context.shape[0])]
        else:
            pipeline_input = original_context

        forecast = self.pipeline.predict(pipeline_input, prediction_length=prediction_length)

        # 5. Reset weights and requires_grad state
        restore_layers(saved)
        for param in model.parameters():
            param.requires_grad_(True)

        return forecast
