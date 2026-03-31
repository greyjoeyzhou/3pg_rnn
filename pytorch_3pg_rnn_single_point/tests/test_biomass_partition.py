"""Tests for BiomassPartition.py — allocation, litterfall, and d13C."""

import torch
import pytest

from BiomassPartition import (
    calc_canopy_conductance,
    calc_biomass_partition,
    calc_litter_and_rootturnover,
    update_endofmonth_biomass,
    calc_d13c,
)


def T(val):
    return torch.tensor([[val]], dtype=torch.float32)


# ---- Canopy conductance --------------------------------------------------

class TestCanopyConductance:
    def test_positive_output(self):
        gc = calc_canopy_conductance(
            T_av=T(15), LAI=T(3), modifier_physiology=T(0.8),
            TK2=T(0.244), TK3=T(0.0368), MaxCond=T(0.0135), LAIgcx=T(3.33),
        )
        assert gc.item() > 0

    def test_minimum_floor(self):
        """Even with zero LAI, conductance should be >= 0.0001."""
        gc = calc_canopy_conductance(
            T_av=T(15), LAI=T(0), modifier_physiology=T(0.8),
            TK2=T(0.244), TK3=T(0.0368), MaxCond=T(0.0135), LAIgcx=T(3.33),
        )
        assert gc.item() == pytest.approx(0.0001, abs=1e-6)

    def test_increases_with_LAI(self):
        gc_low = calc_canopy_conductance(
            T(15), T(1), T(0.8), T(0.244), T(0.0368), T(0.0135), T(3.33),
        ).item()
        gc_high = calc_canopy_conductance(
            T(15), T(3), T(0.8), T(0.244), T(0.0368), T(0.0135), T(3.33),
        ).item()
        assert gc_high > gc_low


# ---- Biomass partitioning ------------------------------------------------

class TestBiomassPartition:
    def test_partitions_sum_to_npp(self):
        NPP = T(2.0)
        delWF, delWR, delWS = calc_biomass_partition(
            NPP=NPP, avDBH=T(10), modifier_physiology=T(0.8),
            m0=T(0), FR=T(0.5), pfsConst=T(0.7), pfsPower=T(-0.38),
            pRx=T(0.45), pRn=T(0.25),
        )
        total = delWF.item() + delWR.item() + delWS.item()
        assert total == pytest.approx(NPP.item(), rel=1e-4)

    def test_all_partitions_positive(self):
        delWF, delWR, delWS = calc_biomass_partition(
            NPP=T(2.0), avDBH=T(10), modifier_physiology=T(0.8),
            m0=T(0), FR=T(0.5), pfsConst=T(0.7), pfsPower=T(-0.38),
            pRx=T(0.45), pRn=T(0.25),
        )
        assert delWF.item() > 0
        assert delWR.item() > 0
        assert delWS.item() > 0

    def test_more_roots_under_stress(self):
        """Lower physiological modifier -> higher root fraction."""
        _, delWR_good, _ = calc_biomass_partition(
            T(2.0), T(10), T(0.9), T(0), T(0.5), T(0.7), T(-0.38), T(0.45), T(0.25),
        )
        _, delWR_bad, _ = calc_biomass_partition(
            T(2.0), T(10), T(0.2), T(0), T(0.5), T(0.7), T(-0.38), T(0.45), T(0.25),
        )
        assert delWR_bad.item() > delWR_good.item()


# ---- Litterfall and root turnover ----------------------------------------

class TestLitterAndRootturnover:
    def test_positive_values(self):
        delLitter, delRoots = calc_litter_and_rootturnover(
            WF=T(5.0), WR=T(3.0), stand_age=T(30),
            gammaFx=T(0.011), gammaF0=T(0.001), tgammaF=T(24), Rttover=T(0.015),
        )
        assert delLitter.item() > 0
        assert delRoots.item() > 0

    def test_litterfall_increases_with_age(self):
        young, _ = calc_litter_and_rootturnover(
            T(5.0), T(3.0), T(5), T(0.011), T(0.001), T(24), T(0.015),
        )
        old, _ = calc_litter_and_rootturnover(
            T(5.0), T(3.0), T(60), T(0.011), T(0.001), T(24), T(0.015),
        )
        assert old.item() > young.item()


# ---- End-of-month biomass update -----------------------------------------

class TestUpdateBiomass:
    def test_mass_balance(self):
        """Total biomass change = NPP - litter - root turnover."""
        WF0, WR0, WS0 = T(5.0), T(3.0), T(10.0)
        TL0 = T(1.0)
        delWF, delWR, delWS = T(0.5), T(0.3), T(0.8)
        delLitter, delRoots = T(0.1), T(0.05)

        WF, WR, WS, TotalW, TotalLitter = update_endofmonth_biomass(
            WF0, WR0, WS0, TL0, delWF, delWR, delWS, delLitter, delRoots,
        )

        expected_total = (WF0 + WR0 + WS0).item() + (
            delWF + delWR + delWS - delLitter - delRoots
        ).item()
        assert TotalW.item() == pytest.approx(expected_total, rel=1e-4)

    def test_litter_accumulates(self):
        _, _, _, _, TotalLitter = update_endofmonth_biomass(
            T(5), T(3), T(10), T(1.0), T(0.5), T(0.3), T(0.8), T(0.1), T(0.05),
        )
        assert TotalLitter.item() == pytest.approx(1.1, rel=1e-4)


# ---- d13C ----------------------------------------------------------------

class TestCalcD13C:
    def test_reasonable_range(self):
        """Tree-ring d13C should typically be between -35 and -20 per mille."""
        d13c, ci = calc_d13c(
            T_av=T(15), CaMonthly=T(400), D13Catm=T(-8.0),
            elev=T(500), GPPmolc=T(10.0), days_in_month=T(30),
            canopy_conductance=T(0.01),
            RGcGW=T(0.66), D13CTissueDif=T(1.99),
            aFracDiffu=T(4.4), bFracRubi=T(27),
        )
        assert -35 < d13c.item() < -15

    def test_ci_positive(self):
        _, ci = calc_d13c(
            T(15), T(400), T(-8.0), T(500), T(10.0), T(30),
            T(0.01), T(0.66), T(1.99), T(4.4), T(27),
        )
        assert ci.item() > 0

    def test_gradient_flows(self):
        """Ensure autograd works through d13c calculation."""
        gc = torch.tensor([[0.01]], requires_grad=True)
        d13c, _ = calc_d13c(
            T(15), T(400), T(-8.0), T(500), T(10.0), T(30),
            gc, T(0.66), T(1.99), T(4.4), T(27),
        )
        d13c.backward()
        assert gc.grad is not None
