"""Shared PyTorch compatibility utilities.

Provides consistent aliases and helper functions used across all model modules,
avoiding duplication of torch wrapper functions.
"""

import torch

# Torch function aliases for cleaner mathematical notation
exp = torch.exp
clip = torch.clamp
minimum = torch.min
maximum = torch.max


def log(x):
    """Natural logarithm that handles both tensor and scalar inputs.

    Unlike a naive ``torch.log(torch.tensor(x))`` wrapper, this preserves
    gradient flow when *x* is already a tensor.
    """
    if isinstance(x, torch.Tensor):
        return torch.log(x)
    return torch.log(torch.tensor(float(x)))


def concatenate(list_x):
    """Concatenate tensors along the last dimension."""
    return torch.cat(list_x, dim=-1)


def where(condition, x, y):
    """Element-wise conditional selection."""
    return torch.where(condition, x, y)


def cast(tensor, dtype_name):
    """Cast a tensor to the named PyTorch dtype (e.g. ``'float32'``).

    Raises:
        ValueError: If *dtype_name* is not a valid PyTorch dtype.
    """
    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        raise ValueError(f"Data type '{dtype_name}' not recognized by PyTorch")
    return tensor.to(dtype)
