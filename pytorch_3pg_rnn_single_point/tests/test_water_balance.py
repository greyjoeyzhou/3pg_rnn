"""Tests for WaterBalance.py — transpiration, interception, and soil water."""

import torch
import pytest

from WaterBalance import (
    calc_transpiration_PM,
    calc_interception,
    calc_soil_water_balance,
)


def T(val):
    return torch.tensor([[val]], dtype=torch.float32)


# ---- Transpiration (Penman-Monteith) --------------------------------------

class TestTranspirationPM:
    def test_positive_transpiration(self):
        tr = calc_transpiration_PM(
            Q=T(15), VPD=T(1.0), h=T(43200), gBL=T(0.2), gC=T(0.01),
        )
        assert tr.item() > 0

    def test_higher_vpd_more_transpiration(self):
        tr_low = calc_transpiration_PM(T(15), T(0.5), T(43200), T(0.2), T(0.01)).item()
        tr_high = calc_transpiration_PM(T(15), T(2.0), T(43200), T(0.2), T(0.01)).item()
        assert tr_high > tr_low

    def test_gradient_flows(self):
        gC = torch.tensor([[0.01]], requires_grad=True)
        tr = calc_transpiration_PM(T(15), T(1.0), T(43200), T(0.2), gC)
        tr.backward()
        assert gC.grad is not None


# ---- Interception ---------------------------------------------------------

class TestInterception:
    def test_zero_LAI_no_interception(self):
        ic = calc_interception(rain=T(50), LAI=T(0), LAImaxIntcptn=T(0), MaxIntcptn=T(0.189))
        assert ic.item() == pytest.approx(0.0, abs=1e-4)

    def test_high_LAI_max_interception(self):
        ic = calc_interception(T(100), T(5), T(1), T(0.189))
        assert ic.item() == pytest.approx(18.9, rel=1e-2)

    def test_no_rain_no_interception(self):
        ic = calc_interception(T(0), T(3), T(1), T(0.189))
        assert ic.item() == pytest.approx(0.0, abs=1e-6)


# ---- Soil water balance --------------------------------------------------

class TestSoilWaterBalance:
    def test_bounded_by_max(self):
        """ASW should not exceed MaxASW."""
        ASW, _ = calc_soil_water_balance(
            ASW=T(190), rain=T(100), loss_water=T(10), irrig=T(0),
            MinASW=T(0), MaxASW=T(200),
        )
        assert ASW.item() <= 200.0 + 1e-6

    def test_bounded_by_min(self):
        """ASW should not go below MinASW."""
        ASW, _ = calc_soil_water_balance(
            ASW=T(10), rain=T(0), loss_water=T(100), irrig=T(0),
            MinASW=T(0), MaxASW=T(200),
        )
        assert ASW.item() >= -1e-6

    def test_water_balance_arithmetic(self):
        """When within bounds, ASW = ASW + rain - loss."""
        ASW, _ = calc_soil_water_balance(
            T(100), T(50), T(30), T(0), T(0), T(200),
        )
        assert ASW.item() == pytest.approx(120.0, rel=1e-4)
