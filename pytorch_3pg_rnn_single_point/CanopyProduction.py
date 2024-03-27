# %%
"""
Canopy Production Module
"""

# from math import exp
# from utils import get_days_in_month
from const import molPAR_MJ, gDM_mol

# from tensorflow.keras.backend import exp, clip, minimum, concatenate, cast
import torch

exp = torch.exp
clip = torch.clamp
minimum = torch.min


def concatenate(list_x):
    return torch.cat(list_x, dim=-1)


# Cast function (to change tensor data types, similar to NumPy's astype)
# cast = torch.tensor.type
def get_dtype_from_name(dtype_name):
    try:
        dtype = getattr(torch, dtype_name)
        return dtype
    except AttributeError:
        raise ValueError(f"Data type {dtype_name} not recognized by PyTorch")


def cast(ts, str_dtype):
    dtype = get_dtype_from_name(str_dtype)
    return ts.to(dtype)


def calc_modifier_temp(T_av, T_min, T_max, T_opt):
    res = ((T_av - T_min) / (T_opt - T_min)) * ((T_max - T_av) / (T_max - T_opt)) ** (
        (T_max - T_opt) / (T_opt - T_min)
    )
    res = clip(res, 0, 1)
    return res


def calc_modifier_VPD(VPD, CoeffCond):
    res = exp(-1 * CoeffCond * VPD)
    return res


def calc_modifier_soilwater(ASW, MaxASW, SWconst, SWpower):
    moist_ratio = ASW / MaxASW
    res = 1 / (1 + ((1 - moist_ratio) / SWconst) ** SWpower)
    return res


def calc_modifier_soilnutrition(FR, fN0):
    res = fN0 + (1 - fN0) * FR
    return res


def calc_modifier_frost(frost_days, kF):
    res = 1 - kF * cast(frost_days / 30, "float32")
    return res


def calc_modifier_age(stand_age, MaxAge, rAge, nAge):
    rel_age = stand_age / MaxAge
    res = 1 / (1 + (rel_age / rAge) ** nAge)
    return res


def calc_physiological_modifier(modifier_VPD, modifier_soilwater, modifier_age):
    # calculate physiological modifier applied to conductance and APARu.
    res = minimum(modifier_VPD, modifier_soilwater) * modifier_age
    return res


def calc_canopy_cover(stand_age, LAI, fullCanAge, canpower, k):
    # calc canopy cover and light interception.
    canopy_cover = (stand_age / fullCanAge) ** canpower
    canopy_cover = clip(canopy_cover, 0.0, 1.0)
    light_interception = 1 - (exp(-1 * k * LAI))

    return canopy_cover, light_interception


def calc_canopy_production(
    solar_rad,
    days_in_month,
    light_interception,
    canopy_cover,
    modifier_physiology,
    modifier_nutrition,
    modifier_temperature,
    modifier_frost,
    alpha,
    y,
):
    # Determine gross and net biomass production
    # Calculate PAR, APAR, APARu and GPP

    RAD = solar_rad * days_in_month  # MJ/m^2
    PAR = RAD * molPAR_MJ  # mol/m^2
    APAR = PAR * light_interception * canopy_cover
    APARu = APAR * modifier_physiology
    alphaC = alpha * modifier_nutrition * modifier_temperature * modifier_frost
    GPPmolc = APARu * alphaC  # mol/m^2
    GPPdm = (GPPmolc * gDM_mol) / 100  # tDM/ha
    NPP = GPPdm * y  # assumes constant respiratory rate

    return PAR, APAR, APARu, GPPmolc, GPPdm, NPP


def canopy_production(
    T_av,
    VPD,
    ASW,
    frost_days,
    stand_age,
    LAI,
    solar_rad,
    days_in_month,
    CounterforShrub,
    paras,
    site_paras,
):
    T_min = paras.T_min
    T_max = paras.T_max
    T_opt = paras.T_opt

    CoeffCond = paras.CoeffCond

    MaxASW = site_paras.MaxASW  # [0]
    SWconst = site_paras.SWconst0  # [2]
    SWpower = site_paras.SWpower0  # [3]

    FR = site_paras.FR  # [4]
    fN0 = paras.fN0

    kF = paras.kF

    MaxAge = site_paras.MaxAge  # [5]
    rAge = paras.rAge
    nAge = paras.nAge

    fullCanAge = paras.fullCanAge
    canpower = paras.canpower
    k = paras.k

    alpha = paras.alpha
    y = paras.y

    #     if CounterforShrub is None:
    CounterforShrub = paras.CounterforShrub

    KL = paras.KL
    Lsx = paras.Lsx

    modifier_temperature = calc_modifier_temp(T_av, T_min, T_max, T_opt)
    modifier_VPD = calc_modifier_VPD(VPD, CoeffCond)
    modifier_soilwater = calc_modifier_soilwater(ASW, MaxASW, SWconst, SWpower)
    modifier_nutrition = calc_modifier_soilnutrition(FR, fN0)

    modifier_frost = calc_modifier_frost(frost_days, kF)
    modifier_age = calc_modifier_age(stand_age, MaxAge, rAge, nAge)
    modifier_physiology = calc_physiological_modifier(
        modifier_VPD, modifier_soilwater, modifier_age
    )

    canopy_cover, light_interception = calc_canopy_cover(
        stand_age, LAI, fullCanAge, canpower, k
    )
    PAR, APAR, APARu, GPPmolc, GPPdm, NPP = calc_canopy_production(
        solar_rad,
        days_in_month,
        light_interception,
        canopy_cover,
        modifier_physiology,
        modifier_nutrition,
        modifier_temperature,
        modifier_frost,
        alpha,
        y,
    )

    list_modifiers = [
        modifier_temperature,
        modifier_VPD,
        modifier_soilwater,
        modifier_nutrition,
        modifier_frost,
        modifier_age,
        modifier_physiology,
    ]
    modifiers = concatenate(list_modifiers)

    # TODO
    LsOpen = LAI * KL
    LsClosed = Lsx * exp(-k * LAI)
    LAIShrub = minimum(LsOpen, LsClosed)

    return PAR, APAR, APARu, GPPmolc, GPPdm, NPP, modifiers, LAIShrub, CounterforShrub


# %%
