"""Tests for torch_compat.py — shared PyTorch wrappers."""

import torch
import pytest

from torch_compat import log, cast, concatenate, where


class TestLog:
    def test_scalar_input(self):
        result = log(2.0)
        assert result.item() == pytest.approx(0.6931, abs=1e-3)

    def test_tensor_input_preserves_grad(self):
        """log() must not break gradient flow for tensor inputs."""
        x = torch.tensor(2.0, requires_grad=True)
        y = log(x)
        y.backward()
        assert x.grad is not None
        assert x.grad.item() == pytest.approx(0.5, abs=1e-4)

    def test_tensor_input_correct_value(self):
        x = torch.tensor(1.0)
        assert log(x).item() == pytest.approx(0.0, abs=1e-6)


class TestCast:
    def test_float32(self):
        t = torch.tensor([1, 2, 3])
        result = cast(t, "float32")
        assert result.dtype == torch.float32

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="not recognized"):
            cast(torch.tensor([1.0]), "invalid_type")


class TestConcatenate:
    def test_basic(self):
        a = torch.tensor([[1.0, 2.0]])
        b = torch.tensor([[3.0, 4.0]])
        result = concatenate([a, b])
        assert result.shape == (1, 4)
        assert result[0, 2].item() == 3.0


class TestWhere:
    def test_basic(self):
        cond = torch.tensor([True, False, True])
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([4.0, 5.0, 6.0])
        result = where(cond, x, y)
        assert result[0].item() == 1.0
        assert result[1].item() == 5.0
