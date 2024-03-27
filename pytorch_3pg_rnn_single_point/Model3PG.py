from CanopyProduction import canopy_production
from BiomassPartition import biomass_partion
from WaterBalance import water_balance
from StemMortality import stem_mortality


def run3pg(
    # previous step:
    LAI_prev,
    ASW_prev,
    StemNo_prev,
    stand_age_prev,
    WF_prev,
    WR_prev,
    WS_prev,
    TotalLitter_prev,
    avDBH_prev,
    delStemNo_prev,
    # current input:
    T_av,
    VPD,
    rain,
    solar_rad,
    frost_days,
    CaMonthly,
    D13Catm,
    day_length,
    days_in_month,
    # site parameters:
    site_paras,
    # model parameters:
    model_paras,
):
    CounterforShrub = None  # TODO

    # Canopy Production Module
    (
        PAR,
        APAR,
        APARu,
        GPPmolc,
        GPPdm,
        NPP,
        modifiers,
        LAIShrub,
        CounterforShrub,
    ) = canopy_production(
        T_av,
        VPD,
        ASW_prev,
        frost_days,
        stand_age_prev,
        LAI_prev,
        solar_rad,
        days_in_month,
        CounterforShrub,
        model_paras,
        site_paras,
    )

    # Biomass Partion Module
    modifier_physiology = modifiers[:, -1:]

    (
        WF,
        WR,
        WS,
        TotalW,
        TotalLitter,
        D13CTissue,
        InterCiPPM,
        canopy_conductance,
    ) = biomass_partion(
        T_av,
        LAI_prev,
        site_paras.elev,
        CaMonthly,
        D13Catm,
        WF_prev,
        WR_prev,
        WS_prev,
        TotalLitter_prev,
        NPP,
        GPPmolc,
        stand_age_prev,
        days_in_month,
        avDBH_prev,
        modifier_physiology,
        model_paras,
        site_paras,
    )

    # Water Balance Module
    irrig = 0
    transpall, transp, transpshrub, loss_water, ASW, monthlyIrrig = water_balance(
        solar_rad,
        VPD,
        day_length,
        LAI_prev,
        rain,
        irrig,
        days_in_month,
        ASW_prev,
        canopy_conductance,
        LAIShrub,
        model_paras,
        site_paras,
    )

    # Stem Mortaility Module
    (
        stand_age,
        LAI,
        MAI,
        avDBH,
        BasArea,
        Height,
        StemNo,
        delStemNo,
        StandVol,
        WF,
        WR,
        WS,
    ) = stem_mortality(
        WF, WR, WS, StemNo_prev, delStemNo_prev, stand_age_prev, model_paras
    )

    list_out = [
        StandVol,
        LAI,
        ASW,
        StemNo,
        PAR,
        stand_age,
        WF,
        WR,
        WS,
        TotalLitter,
        avDBH,
        delStemNo,
        D13CTissue,
    ]
    return list_out


def calc_factors_age_np(stand_age, SLA0, SLA1, tSLA, fracBB0, fracBB1, tBB):
    # update age-dependent factors
    import numpy as np

    SLA = SLA1 + (SLA0 - SLA1) * np.exp(-np.log(2.0) * (stand_age / tSLA) ** 2)
    fracBB = fracBB1 + (fracBB0 - fracBB1) * np.exp(-np.log(2.0) * (stand_age / tBB))
    return SLA, fracBB


# prepare data required to run model
def prepare(
    fpath_setting, list_trainable=["alpha", "MaxCond", "CoeffCond", "fullCanAge"]
):
    import os
    import numpy as np
    import pandas as pd
    from yaml import load

    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    from utils import get_stand_age, get_day_length, get_days_in_month

    # from utils import get_VPD
    import parameters as paras

    sett = load(open(fpath_setting), Loader=Loader)

    sett_time = sett["time_range"]
    sett_initial = sett["initial_state"]
    site_paras = sett["site_paras"]

    mYears = sett_time["EndYear"] - sett_time["InitialYear"] + 1
    # Assign initial state of stand
    stand_age, StartAge, InitialYear, InitialMonth, MonthPlanted = get_stand_age(
        site_paras["lat"],
        sett_time["InitialYear"],
        sett_time["InitialMonth"],
        sett_time["YearPlanted"],
        sett_time["MonthPlanted"],
        sett_time["EndAge"],
    )

    WS = sett_initial["InitialWS"]
    WF = sett_initial["InitialWF"]
    WR = sett_initial["InitialWR"]
    StemNo = sett_initial["InitialStocking"]
    ASW = sett_initial["InitialASW"]
    # thinEventNo = 1
    # defoltnEventNo = 1

    SLA, fracBB = calc_factors_age_np(
        stand_age,
        paras.SLA0,
        paras.SLA1,
        paras.tSLA,
        paras.fracBB0,
        paras.fracBB1,
        paras.tBB,
    )
    AvStemMass = WS * 1000 / StemNo  # kg/tree
    avDBH = (AvStemMass / paras.StemConst) ** (1 / paras.StemPower)
    LAI = WF * SLA * 0.1
    StandVol = WS * (1 - fracBB) / paras.Density

    TotalLitter = 0

    delStemNo = 0

    PAR = 0

    D13CTissue = -32.0

    initial = [
        StandVol,
        LAI,
        ASW,
        StemNo,
        PAR,
        stand_age,
        WF,
        WR,
        WS,
        TotalLitter,
        avDBH,
        delStemNo,
        D13CTissue,
    ]

    arr_day_length = []
    arr_days_in_month = []

    # do annual calculation
    metMonth = InitialMonth
    for year in range(StartAge, sett_time["EndAge"] + 1):
        # print('year', year)

        # do monthly calculations
        month = InitialMonth
        for month_counter in range(1, 12 + 1):
            if month >= 12:
                month = 1
            arr_day_length.append(get_day_length(site_paras["lat"], month))
            arr_days_in_month.append(get_days_in_month(month))
            metMonth = metMonth + 1
            month = month + 1
    arr_day_length = np.array(arr_day_length).reshape((-1, 1))
    arr_days_in_month = np.array(arr_days_in_month).reshape((-1, 1))

    arr_site_paras = np.array(
        [
            site_paras["MaxASW"],
            site_paras["MinASW"],
            site_paras["SWconst0"],
            site_paras["SWpower0"],
            site_paras["FR"],
            site_paras["MaxAge"],
            site_paras["elev"],
        ]
    )

    # load input time series data
    if "fpath_input" in sett["io"] and sett["io"]["fpath_input"] is not None:
        df_input = pd.read_csv(
            os.path.join(os.path.dirname(fpath_setting), sett["io"]["fpath_input"]),
            delimiter="\t",
        )
        arr_input = df_input.values
    else:
        arr_input = None

    # load target data
    if "fpath_target" in sett["io"] and sett["io"]["fpath_target"] is not None:
        df_target = pd.read_csv(
            os.path.join(os.path.dirname(fpath_setting), sett["io"]["fpath_target"]),
            delimiter="\t",
        )
        arr_target = df_target[["treering", "D13C"]].values
    else:
        arr_target = None

    dict_initial_weights = {}
    for para in dir(paras):
        if not (para.startswith("__") or para == "log"):
            dict_initial_weights[para] = {}
            dict_initial_weights[para]["value"] = getattr(paras, para)
            if para in list_trainable:
                dict_initial_weights[para]["trainable"] = True
            else:
                dict_initial_weights[para]["trainable"] = False
    # overwrite default
    for para in sett["paras"]:
        dict_initial_weights[para]["value"] = sett["paras"][para]

    return (
        initial,
        arr_day_length,
        arr_days_in_month,
        arr_site_paras,
        arr_input,
        arr_target,
        dict_initial_weights,
    )
