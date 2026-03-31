"""Tests for StemMortality.py — self-thinning, age factors, stand metrics."""

import torch
import pytest
from math import pi

from StemMortality import (
    getMortality,
    calc_mortality,
    calc_factors_age,
    update_stands,
)


def T(val):
    return torch.tensor([[val]], dtype=torch.float32)


# ---- getMortality ---------------------------------------------------------

class TestGetMortality:
    def test_returns_finite(self):
        """getMortality returns the number of stems to remove (can be negative
        when the stand is far from the self-thinning boundary)."""
        res = getMortality(
            oldN=T(500), oldW=T(50), mS=T(0.2), wSx1000=T(245), thinPower=T(1.5),
        )
        assert torch.isfinite(res)

    def test_high_wsx_means_more_removal_needed(self):
        """Higher wSx1000 should allow larger trees, reducing mortality."""
        res_low = getMortality(T(500), T(50), T(0.2), T(100), T(1.5))
        res_high = getMortality(T(500), T(50), T(0.2), T(500), T(1.5))
        # Higher wSx1000 -> fewer stems need removal
        assert res_high.item() < res_low.item()


# ---- calc_mortality -------------------------------------------------------

class TestCalcMortality:
    def test_no_mortality_when_small(self):
        """No thinning when avg stem mass is below maximum."""
        WF, WR, WS, _, StemNo, delStemNo = calc_mortality(
            WF=T(2), WR=T(1), WS=T(5), StemNo=T(500), delStemNo=T(0),
            wSx1000=T(3000), thinPower=T(1.5), mF=T(0), mR=T(0.2), mS=T(0.2),
        )
        # wSx1000=3000 is very high -> no mortality expected
        assert StemNo.item() == pytest.approx(500.0, abs=0.5)

    def test_biomass_conserved_direction(self):
        """After mortality, stem biomass should not increase."""
        WS_init = T(50)
        _, _, WS_out, _, _, _ = calc_mortality(
            T(5), T(3), WS_init, T(500), T(0),
            T(245), T(1.5), T(0), T(0.2), T(0.2),
        )
        assert WS_out.item() <= WS_init.item() + 1e-6


# ---- calc_factors_age -----------------------------------------------------

class TestCalcFactorsAge:
    def test_young_stand_near_SLA0(self):
        SLA, _ = calc_factors_age(
            stand_age=T(0.1), SLA0=T(6.0), SLA1=T(4.0),
            tSLA=T(2.5), fracBB0=T(0.15), fracBB1=T(0.15), tBB=T(1.5),
        )
        assert SLA.item() == pytest.approx(6.0, abs=0.1)

    def test_old_stand_near_SLA1(self):
        SLA, _ = calc_factors_age(
            T(100), T(6.0), T(4.0), T(2.5), T(0.15), T(0.15), T(1.5),
        )
        assert SLA.item() == pytest.approx(4.0, abs=0.1)

    def test_fracBB_converges(self):
        """fracBB should converge to fracBB1 for old stands."""
        _, fracBB = calc_factors_age(
            T(100), T(6.0), T(4.0), T(2.5), T(0.20), T(0.10), T(1.5),
        )
        assert fracBB.item() == pytest.approx(0.10, abs=0.01)


# ---- update_stands --------------------------------------------------------

class TestUpdateStands:
    def test_lai_positive(self):
        LAI, *_ = update_stands(
            stand_age=T(20), WF=T(3), WS=T(30), AvStemMass=T(100),
            StemNo=T(300), SLA=T(5.0), fracBB=T(0.15),
            StemConst=T(0.058), StemPower=T(2.55), Density=T(0.38),
            HtC0=T(5.0), HtC1=T(-8.19),
        )
        assert LAI.item() > 0

    def test_dbh_positive(self):
        _, _, avDBH, *_ = update_stands(
            T(20), T(3), T(30), T(100), T(300), T(5.0), T(0.15),
            T(0.058), T(2.55), T(0.38), T(5.0), T(-8.19),
        )
        assert avDBH.item() > 0

    def test_volume_positive(self):
        _, _, _, _, _, StandVol = update_stands(
            T(20), T(3), T(30), T(100), T(300), T(5.0), T(0.15),
            T(0.058), T(2.55), T(0.38), T(5.0), T(-8.19),
        )
        assert StandVol.item() > 0

    def test_height_positive(self):
        _, _, _, _, Height, _ = update_stands(
            T(20), T(3), T(30), T(100), T(300), T(5.0), T(0.15),
            T(0.058), T(2.55), T(0.38), T(5.0), T(-8.19),
        )
        assert Height.item() > 0
