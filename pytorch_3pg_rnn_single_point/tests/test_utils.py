"""Tests for utils.py — calendar and astronomical helpers."""

import pytest
from utils import get_day_length, get_days_in_month, get_VPD, get_stand_age


class TestGetDayLength:
    def test_equator_roughly_12h(self):
        """Near the equator, day length should be close to 12 hours."""
        for month in range(12):
            dl = get_day_length(0.0, month)
            assert 40000 < dl < 46000, f"month={month}, dl={dl}"

    def test_northern_summer_longer(self):
        """Northern mid-latitudes: June (5) > December (11)."""
        june = get_day_length(45.0, 5)
        december = get_day_length(45.0, 11)
        assert june > december

    def test_southern_hemisphere_opposite(self):
        """Southern hemisphere: December longer than June."""
        june = get_day_length(-45.0, 5)
        december = get_day_length(-45.0, 11)
        assert december > june

    def test_positive_for_all_months(self):
        for month in range(12):
            assert get_day_length(47.0, month) > 0

    def test_high_latitude_polar_day(self):
        """Very high latitude in summer can approach 24h."""
        dl = get_day_length(80.0, 5)  # June at 80N
        assert dl > 70000  # > ~19.4 hours

    def test_returns_float(self):
        assert isinstance(get_day_length(10.0, 0), float)


class TestGetDaysInMonth:
    def test_january(self):
        assert get_days_in_month(0) == 31

    def test_february(self):
        assert get_days_in_month(1) == 28

    def test_april(self):
        assert get_days_in_month(3) == 30

    def test_all_months_sum_to_365(self):
        total = sum(get_days_in_month(m) for m in range(12))
        assert total == 365


class TestGetVPD:
    def test_positive_when_tmax_gt_tmin(self):
        vpd = get_VPD(10, 25)
        assert vpd > 0

    def test_zero_when_equal(self):
        vpd = get_VPD(20, 20)
        assert vpd == pytest.approx(0.0, abs=1e-6)

    def test_increases_with_temperature_range(self):
        vpd_narrow = get_VPD(15, 20)
        vpd_wide = get_VPD(10, 25)
        assert vpd_wide > vpd_narrow


class TestGetStandAge:
    def test_basic(self):
        age, start, *_ = get_stand_age(45, 1950, 1, 1930, 1, 100)
        assert age == pytest.approx(20.0)
        assert start == 20

    def test_negative_age_raises(self):
        """When InitialYear < YearPlanted, the code adds them, which can
        exceed EndAge.  Either 'must be >= 0' or 'exceeds' is raised."""
        with pytest.raises(Exception):
            get_stand_age(45, 1920, 1, 1930, 1, 100)

    def test_start_exceeds_end_raises(self):
        with pytest.raises(Exception, match="exceeds the ending age"):
            get_stand_age(45, 1950, 1, 1930, 1, 10)

    def test_auto_month_northern(self):
        """InitialMonth=0 for northern lat defaults to 0."""
        _, _, _, init_month, _ = get_stand_age(45, 1950, 0, 1930, 0, 100)
        assert init_month == 0

    def test_auto_month_southern(self):
        """InitialMonth=0 for southern lat defaults to 6."""
        _, _, _, init_month, _ = get_stand_age(-45, 1950, 0, 1930, 0, 100)
        assert init_month == 6
