"""Tests for Model3PG.py — orchestration and data preparation helpers."""

import os
import pytest
import numpy as np

from Model3PG import (
    calc_factors_age_np,
    _load_config,
    _compute_initial_state,
    _build_parameter_dict,
    prepare,
)


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data_files")
CONFIG_PATH = os.path.join(DATA_DIR, "exp.yaml")
HAS_DATA = os.path.exists(CONFIG_PATH)


# ---- calc_factors_age_np --------------------------------------------------

class TestCalcFactorsAgeNp:
    def test_young_near_SLA0(self):
        SLA, _ = calc_factors_age_np(0.1, 6.0, 4.0, 2.5, 0.15, 0.15, 1.5)
        assert abs(SLA - 6.0) < 0.15

    def test_old_near_SLA1(self):
        SLA, _ = calc_factors_age_np(100, 6.0, 4.0, 2.5, 0.15, 0.15, 1.5)
        assert abs(SLA - 4.0) < 0.1

    def test_returns_tuple(self):
        result = calc_factors_age_np(10, 5.6, 5.6, 2.5, 0.15, 0.15, 1.5)
        assert len(result) == 2


# ---- Data preparation (requires data files) -------------------------------

@pytest.mark.skipif(not HAS_DATA, reason="data files not present")
class TestPrepare:
    def test_returns_seven_elements(self):
        result = prepare(CONFIG_PATH)
        assert len(result) == 7

    def test_initial_state_length(self):
        initial, *_ = prepare(CONFIG_PATH)
        assert len(initial) == 13

    def test_day_length_array_shape(self):
        _, arr_dl, arr_dim, *_ = prepare(CONFIG_PATH)
        assert arr_dl.ndim == 2
        assert arr_dl.shape[1] == 1
        assert arr_dim.shape == arr_dl.shape

    def test_input_data_loaded(self):
        _, _, _, _, arr_input, _, _ = prepare(CONFIG_PATH)
        assert arr_input is not None
        assert arr_input.ndim == 2

    def test_target_data_loaded(self):
        _, _, _, _, _, arr_target, _ = prepare(CONFIG_PATH)
        assert arr_target is not None
        assert arr_target.shape[1] == 2

    def test_parameter_dict_has_alpha(self):
        *_, dict_w = prepare(CONFIG_PATH)
        assert "alpha" in dict_w
        assert dict_w["alpha"]["trainable"] is True

    def test_non_trainable_default(self):
        *_, dict_w = prepare(CONFIG_PATH)
        assert dict_w["T_min"]["trainable"] is False


@pytest.mark.skipif(not HAS_DATA, reason="data files not present")
class TestLoadConfig:
    def test_returns_dict(self):
        sett = _load_config(CONFIG_PATH)
        assert isinstance(sett, dict)
        assert "time_range" in sett
        assert "site_paras" in sett


@pytest.mark.skipif(not HAS_DATA, reason="data files not present")
class TestComputeInitialState:
    def test_initial_state_values(self):
        sett = _load_config(CONFIG_PATH)
        initial, start_age, init_month = _compute_initial_state(sett)
        assert len(initial) == 13
        assert start_age >= 0


@pytest.mark.skipif(not HAS_DATA, reason="data files not present")
class TestBuildParameterDict:
    def test_trainable_flag(self):
        sett = _load_config(CONFIG_PATH)
        d = _build_parameter_dict(sett, ["alpha", "MaxCond"])
        assert d["alpha"]["trainable"] is True
        assert d["MaxCond"]["trainable"] is True
        assert d["T_min"]["trainable"] is False
