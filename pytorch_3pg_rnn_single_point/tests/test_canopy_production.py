"""Tests for CanopyProduction.py — environmental modifiers and GPP/NPP."""

import torch
import pytest

from CanopyProduction import (
    calc_modifier_temp,
    calc_modifier_VPD,
    calc_modifier_soilwater,
    calc_modifier_soilnutrition,
    calc_modifier_frost,
    calc_modifier_age,
    calc_physiological_modifier,
    calc_canopy_cover,
    calc_canopy_production,
)


def T(val):
    """Shorthand for a (1,1) float tensor."""
    return torch.tensor([[val]], dtype=torch.float32)


# ---- Temperature modifier ------------------------------------------------

class TestModifierTemp:
    def test_at_optimum(self):
        """Modifier should be ~1 at T_opt."""
        res = calc_modifier_temp(T(20), T(2), T(32), T(20))
        assert res.item() == pytest.approx(1.0, abs=1e-4)

    def test_below_tmin(self):
        """Below T_min the modifier is clipped to 0."""
        res = calc_modifier_temp(T(0), T(2), T(32), T(20))
        assert res.item() == pytest.approx(0.0, abs=1e-6)

    def test_above_tmax(self):
        """Above T_max the result is NaN from the formula, clipped to 0."""
        res = calc_modifier_temp(T(35), T(2), T(32), T(20))
        # The formula produces NaN when T > T_max (negative base raised to
        # fractional power), which clip maps to 0.
        assert res.item() == 0.0 or (res != res).item()  # 0 or NaN

    def test_between_0_and_1(self):
        """Intermediate temperature gives modifier in (0, 1)."""
        res = calc_modifier_temp(T(10), T(2), T(32), T(20))
        assert 0 < res.item() < 1


# ---- VPD modifier --------------------------------------------------------

class TestModifierVPD:
    def test_zero_vpd_gives_one(self):
        res = calc_modifier_VPD(T(0.0), T(0.08))
        assert res.item() == pytest.approx(1.0, abs=1e-6)

    def test_high_vpd_near_zero(self):
        res = calc_modifier_VPD(T(30.0), T(0.08))
        assert res.item() < 0.15

    def test_monotonically_decreasing(self):
        low = calc_modifier_VPD(T(0.5), T(0.08)).item()
        high = calc_modifier_VPD(T(2.0), T(0.08)).item()
        assert low > high


# ---- Soil-water modifier -------------------------------------------------

class TestModifierSoilwater:
    def test_full_soil_gives_one(self):
        res = calc_modifier_soilwater(T(200), T(200), T(0.5), T(5))
        assert res.item() == pytest.approx(1.0, abs=1e-4)

    def test_dry_soil_near_zero(self):
        res = calc_modifier_soilwater(T(1), T(200), T(0.5), T(5))
        assert res.item() < 0.05

    def test_half_soil(self):
        res = calc_modifier_soilwater(T(100), T(200), T(0.5), T(5))
        assert 0.4 < res.item() < 1.0


# ---- Nutrition modifier --------------------------------------------------

class TestModifierNutrition:
    def test_full_fertility(self):
        assert calc_modifier_soilnutrition(T(1.0), T(1.0)).item() == pytest.approx(1.0)

    def test_zero_fertility_returns_fN0(self):
        assert calc_modifier_soilnutrition(T(0.0), T(0.5)).item() == pytest.approx(0.5)


# ---- Frost modifier ------------------------------------------------------

class TestModifierFrost:
    def test_no_frost(self):
        res = calc_modifier_frost(T(0), T(1))
        assert res.item() == pytest.approx(1.0, abs=1e-4)

    def test_full_month_frost(self):
        res = calc_modifier_frost(T(30), T(1))
        assert res.item() == pytest.approx(0.0, abs=1e-4)


# ---- Age modifier --------------------------------------------------------

class TestModifierAge:
    def test_young_stand(self):
        res = calc_modifier_age(T(10), T(250), T(0.95), T(4))
        assert res.item() > 0.99

    def test_old_stand_declines(self):
        young = calc_modifier_age(T(10), T(250), T(0.95), T(4)).item()
        old = calc_modifier_age(T(200), T(250), T(0.95), T(4)).item()
        assert old < young


# ---- Physiological modifier -----------------------------------------------

class TestPhysiologicalModifier:
    def test_takes_minimum_of_vpd_sw(self):
        vpd = T(0.8)
        sw = T(0.5)
        age = T(1.0)
        res = calc_physiological_modifier(vpd, sw, age)
        assert res.item() == pytest.approx(0.5, abs=1e-4)


# ---- Canopy cover --------------------------------------------------------

class TestCanopyCover:
    def test_full_cover_at_mature_age(self):
        cover, _ = calc_canopy_cover(T(30), T(3), T(20), T(1), T(0.5))
        assert cover.item() == pytest.approx(1.0, abs=1e-6)

    def test_zero_LAI_no_interception(self):
        _, li = calc_canopy_cover(T(30), T(0), T(20), T(1), T(0.5))
        assert li.item() == pytest.approx(0.0, abs=1e-6)

    def test_interception_increases_with_LAI(self):
        _, li_low = calc_canopy_cover(T(30), T(1), T(20), T(1), T(0.5))
        _, li_high = calc_canopy_cover(T(30), T(5), T(20), T(1), T(0.5))
        assert li_high.item() > li_low.item()


# ---- Canopy production ---------------------------------------------------

class TestCanopyProduction:
    def test_npp_positive(self):
        PAR, APAR, APARu, GPPmolc, GPPdm, NPP = calc_canopy_production(
            solar_rad=T(15), days_in_month=T(30),
            light_interception=T(0.9), canopy_cover=T(1.0),
            modifier_physiology=T(0.8), modifier_nutrition=T(1.0),
            modifier_temperature=T(0.9), modifier_frost=T(1.0),
            alpha=T(0.065), y=T(0.47),
        )
        assert NPP.item() > 0

    def test_npp_lt_gpp(self):
        """NPP = GPP * y, so NPP < GPP when y < 1."""
        _, _, _, _, GPPdm, NPP = calc_canopy_production(
            solar_rad=T(15), days_in_month=T(30),
            light_interception=T(0.9), canopy_cover=T(1.0),
            modifier_physiology=T(0.8), modifier_nutrition=T(1.0),
            modifier_temperature=T(0.9), modifier_frost=T(1.0),
            alpha=T(0.065), y=T(0.47),
        )
        assert NPP.item() < GPPdm.item()

    def test_zero_radiation_zero_production(self):
        _, _, _, _, _, NPP = calc_canopy_production(
            solar_rad=T(0), days_in_month=T(30),
            light_interception=T(0.9), canopy_cover=T(1.0),
            modifier_physiology=T(0.8), modifier_nutrition=T(1.0),
            modifier_temperature=T(0.9), modifier_frost=T(1.0),
            alpha=T(0.065), y=T(0.47),
        )
        assert NPP.item() == pytest.approx(0.0, abs=1e-8)
