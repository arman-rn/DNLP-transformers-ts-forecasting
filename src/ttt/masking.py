import torch


def create_mask(context, n_mask):
    """Create a mask tensor where last n_mask values are 0 (masked), rest are 1.

    Handles both single tensors (1D) and batched tensors (2D).

    Args:
        context: tensor of shape (context_length,) or (batch_size, context_length)
        n_mask: number of values to mask from the end

    Returns:
        mask tensor of same shape, with last n_mask values set to 0
    """
    mask = torch.ones_like(context)
    mask[..., -n_mask:] = 0
    return mask
