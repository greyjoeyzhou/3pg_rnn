"""3-PG Model Orchestration.

Coordinates the four sub-modules (Canopy Production, Biomass Partition,
Water Balance, Stem Mortality) to advance the model state by one month,
and provides data-loading / initialisation helpers.

References:
    Landsberg & Waring (1997), Forest Ecology and Management 95, 209-228.
"""

import os

import numpy as np
import pandas as pd
from yaml import load

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from CanopyProduction import canopy_production
from BiomassPartition import biomass_partition
from WaterBalance import water_balance
from StemMortality import stem_mortality
from utils import get_stand_age, get_day_length, get_days_in_month
import parameters as paras


def run3pg(
    # previous step:
    LAI_prev, ASW_prev, StemNo_prev, stand_age_prev,
    WF_prev, WR_prev, WS_prev, TotalLitter_prev,
    avDBH_prev, delStemNo_prev,
    # current input:
    T_av, VPD, rain, solar_rad, frost_days,
    CaMonthly, D13Catm, day_length, days_in_month,
    # site parameters:
    site_paras,
    # model parameters:
    model_paras,
):
    """Advance the 3-PG model by one monthly time step.

    Args:
        LAI_prev .. delStemNo_prev: State variables from the previous month.
        T_av .. days_in_month: Meteorological forcing for the current month.
        site_paras: Site-level parameters (SiteParams instance).
        model_paras: Model parameters (RNN_3PG or equivalent).

    Returns:
        List of 13 tensors representing the updated state vector.
    """
    CounterforShrub = None  # TODO: shrub counter logic

    # 1. Canopy Production
    (
        PAR, APAR, APARu, GPPmolc, GPPdm, NPP,
        modifiers, LAIShrub, CounterforShrub,
    ) = canopy_production(
        T_av, VPD, ASW_prev, frost_days, stand_age_prev,
        LAI_prev, solar_rad, days_in_month, CounterforShrub,
        model_paras, site_paras,
    )

    # 2. Biomass Partition
    modifier_physiology = modifiers[:, -1:]

    (
        WF, WR, WS, TotalW, TotalLitter,
        D13CTissue, InterCiPPM, canopy_conductance,
    ) = biomass_partition(
        T_av, LAI_prev, site_paras.elev, CaMonthly, D13Catm,
        WF_prev, WR_prev, WS_prev, TotalLitter_prev,
        NPP, GPPmolc, stand_age_prev, days_in_month, avDBH_prev,
        modifier_physiology, model_paras, site_paras,
    )

    # 3. Water Balance
    irrig = 0
    transpall, transp, transpshrub, loss_water, ASW, monthlyIrrig = water_balance(
        solar_rad, VPD, day_length, LAI_prev, rain, irrig,
        days_in_month, ASW_prev, canopy_conductance, LAIShrub,
        model_paras, site_paras,
    )

    # 4. Stem Mortality
    (
        stand_age, LAI, MAI, avDBH, BasArea, Height,
        StemNo, delStemNo, StandVol, WF, WR, WS,
    ) = stem_mortality(
        WF, WR, WS, StemNo_prev, delStemNo_prev,
        stand_age_prev, model_paras,
    )

    return [
        StandVol, LAI, ASW, StemNo, PAR, stand_age,
        WF, WR, WS, TotalLitter, avDBH, delStemNo, D13CTissue,
    ]


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------

def calc_factors_age_np(stand_age, SLA0, SLA1, tSLA, fracBB0, fracBB1, tBB):
    """Numpy version of age-dependent SLA and branch/bark fraction.

    Used during initialisation (before torch tensors exist).
    """
    SLA = SLA1 + (SLA0 - SLA1) * np.exp(-np.log(2.0) * (stand_age / tSLA) ** 2)
    fracBB = fracBB1 + (fracBB0 - fracBB1) * np.exp(-np.log(2.0) * (stand_age / tBB))
    return SLA, fracBB


def _load_config(fpath_setting):
    """Load and return the YAML configuration dict."""
    with open(fpath_setting) as f:
        return load(f, Loader=Loader)


def _compute_initial_state(sett):
    """Derive the 13-element initial state vector from config settings.

    Returns:
        List of 13 floats (same order as the model state vector).
    """
    sett_time = sett["time_range"]
    sett_initial = sett["initial_state"]
    site_params = sett["site_paras"]

    stand_age, StartAge, InitialYear, InitialMonth, MonthPlanted = get_stand_age(
        site_params["lat"],
        sett_time["InitialYear"], sett_time["InitialMonth"],
        sett_time["YearPlanted"], sett_time["MonthPlanted"],
        sett_time["EndAge"],
    )

    WS = sett_initial["InitialWS"]
    WF = sett_initial["InitialWF"]
    WR = sett_initial["InitialWR"]
    StemNo = sett_initial["InitialStocking"]
    ASW = sett_initial["InitialASW"]

    SLA, fracBB = calc_factors_age_np(
        stand_age,
        paras.SLA0, paras.SLA1, paras.tSLA,
        paras.fracBB0, paras.fracBB1, paras.tBB,
    )
    AvStemMass = WS * 1000 / StemNo
    avDBH = (AvStemMass / paras.StemConst) ** (1 / paras.StemPower)
    LAI = WF * SLA * 0.1
    StandVol = WS * (1 - fracBB) / paras.Density

    return (
        [StandVol, LAI, ASW, StemNo, 0, stand_age,
         WF, WR, WS, 0, avDBH, 0, -32.0],
        StartAge, InitialMonth,
    )


def _compute_calendar_arrays(sett, lat, StartAge, InitialMonth):
    """Build day-length and days-in-month arrays for the full simulation.

    Returns:
        (arr_day_length, arr_days_in_month) each shaped (n_months, 1).
    """
    sett_time = sett["time_range"]
    arr_day_length = []
    arr_days_in_month = []

    for year in range(StartAge, sett_time["EndAge"] + 1):
        month = InitialMonth
        for _ in range(12):
            if month >= 12:
                month = 1
            arr_day_length.append(get_day_length(lat, month))
            arr_days_in_month.append(get_days_in_month(month))
            month += 1

    return (
        np.array(arr_day_length).reshape((-1, 1)),
        np.array(arr_days_in_month).reshape((-1, 1)),
    )


def _load_input_data(sett, fpath_setting):
    """Load meteorological input time series from file.

    Returns:
        numpy array of shape (n_months, n_vars), or None.
    """
    io_cfg = sett["io"]
    if io_cfg.get("fpath_input") is not None:
        fpath = os.path.join(os.path.dirname(fpath_setting), io_cfg["fpath_input"])
        return pd.read_csv(fpath, delimiter="\t").values
    return None


def _load_target_data(sett, fpath_setting):
    """Load observed target data (tree rings, d13C) from file.

    Returns:
        numpy array of shape (n_months, 2), or None.
    """
    io_cfg = sett["io"]
    if io_cfg.get("fpath_target") is not None:
        fpath = os.path.join(os.path.dirname(fpath_setting), io_cfg["fpath_target"])
        return pd.read_csv(fpath, delimiter="\t")[["treering", "D13C"]].values
    return None


def _build_parameter_dict(sett, list_trainable):
    """Build the parameter dictionary with trainability flags.

    Reads default values from the ``parameters`` module, marks those
    in *list_trainable* as trainable, and applies any overrides from
    the YAML config.

    Returns:
        dict mapping parameter name -> {"value": float, "trainable": bool}.
    """
    dict_initial_weights = {}
    for para in dir(paras):
        if para.startswith("__") or para == "log":
            continue
        dict_initial_weights[para] = {
            "value": getattr(paras, para),
            "trainable": para in list_trainable,
        }

    # Apply config overrides
    if sett.get("paras"):
        for para in sett["paras"]:
            dict_initial_weights[para]["value"] = sett["paras"][para]

    return dict_initial_weights


def prepare(
    fpath_setting,
    list_trainable=("alpha", "MaxCond", "CoeffCond", "fullCanAge"),
):
    """Load configuration, input data, and build initial model state.

    This is the main entry point for data preparation.  It delegates to
    focused helper functions for each sub-task.

    Args:
        fpath_setting: Path to the YAML configuration file.
        list_trainable: Parameter names to mark as trainable.

    Returns:
        7-tuple: (initial_state, arr_day_length, arr_days_in_month,
                  arr_site_paras, arr_input, arr_target,
                  dict_initial_weights).
    """
    sett = _load_config(fpath_setting)
    site_params = sett["site_paras"]

    initial, StartAge, InitialMonth = _compute_initial_state(sett)

    arr_day_length, arr_days_in_month = _compute_calendar_arrays(
        sett, site_params["lat"], StartAge, InitialMonth,
    )

    arr_site_paras = np.array([
        site_params["MaxASW"], site_params["MinASW"],
        site_params["SWconst0"], site_params["SWpower0"],
        site_params["FR"], site_params["MaxAge"],
        site_params["elev"],
    ])

    arr_input = _load_input_data(sett, fpath_setting)
    arr_target = _load_target_data(sett, fpath_setting)
    dict_initial_weights = _build_parameter_dict(sett, list_trainable)

    return (
        initial, arr_day_length, arr_days_in_month,
        arr_site_paras, arr_input, arr_target,
        dict_initial_weights,
    )
