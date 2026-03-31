"""Shared fixtures for 3-PG model tests."""

import sys
import os

import pytest
import torch

# Ensure the package root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def device():
    """Default compute device for tests."""
    return "cpu"


@pytest.fixture
def scalar_tensor():
    """Factory for creating single-value tensors."""
    def _make(value):
        return torch.tensor([[value]], dtype=torch.float32)
    return _make
