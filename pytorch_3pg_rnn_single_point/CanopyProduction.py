"""Canopy Production Module for the 3-PG model.

Computes gross and net primary production (GPP / NPP) from solar radiation,
environmental modifiers, and stand characteristics.

References:
    Landsberg & Waring (1997), Forest Ecology and Management 95, 209-228.
"""

from const import molPAR_MJ, gDM_mol, GPP_UNIT_CONV
from torch_compat import exp, clip, minimum, concatenate, cast


def calc_modifier_temp(T_av, T_min, T_max, T_opt):
    """Temperature growth modifier (0-1).

    Asymmetric response peaking at T_opt, zero outside [T_min, T_max].
    Equation 2 in Landsberg & Waring (1997).

    Args:
        T_av: Mean monthly temperature (deg C).
        T_min, T_max, T_opt: Cardinal temperatures (deg C).
    """
    res = ((T_av - T_min) / (T_opt - T_min)) * (
        (T_max - T_av) / (T_max - T_opt)
    ) ** ((T_max - T_opt) / (T_opt - T_min))
    return clip(res, 0, 1)


def calc_modifier_VPD(VPD, CoeffCond):
    """Vapour-pressure-deficit modifier (0-1).

    Exponential decline with increasing VPD.

    Args:
        VPD: Vapour pressure deficit (kPa).
        CoeffCond: Stomatal sensitivity to VPD.
    """
    return exp(-1 * CoeffCond * VPD)


def calc_modifier_soilwater(ASW, MaxASW, SWconst, SWpower):
    """Soil-water availability modifier (0-1).

    Sigmoidal response based on relative available soil water.

    Args:
        ASW: Available soil water (mm).
        MaxASW: Maximum available soil water (mm).
        SWconst: Soil-water response constant.
        SWpower: Soil-water response power.
    """
    moist_ratio = ASW / MaxASW
    return 1 / (1 + ((1 - moist_ratio) / SWconst) ** SWpower)


def calc_modifier_soilnutrition(FR, fN0):
    """Soil nutrition modifier (fN0-1).

    Linear interpolation between fN0 (at FR=0) and 1 (at FR=1).

    Args:
        FR: Fertility rating (0-1).
        fN0: Modifier value when FR = 0.
    """
    return fN0 + (1 - fN0) * FR


def calc_modifier_frost(frost_days, kF):
    """Frost modifier (0-1).

    Reduces production proportionally to frost-day fraction of the month.

    Args:
        frost_days: Number of frost days in the month.
        kF: Production days lost per frost day.
    """
    return 1 - kF * cast(frost_days / 30, "float32")


def calc_modifier_age(stand_age, MaxAge, rAge, nAge):
    """Age-related physiological decline modifier (0-1).

    Sigmoidal decline; fAge = 0.5 when stand_age/MaxAge = rAge.

    Args:
        stand_age: Current stand age (years).
        MaxAge: Maximum stand age (years).
        rAge: Relative age for fAge = 0.5.
        nAge: Power controlling steepness.
    """
    rel_age = stand_age / MaxAge
    return 1 / (1 + (rel_age / rAge) ** nAge)


def calc_physiological_modifier(modifier_VPD, modifier_soilwater, modifier_age):
    """Combined physiological modifier applied to conductance and APARu.

    Takes the minimum of VPD and soil-water modifiers, then multiplies
    by the age modifier.
    """
    return minimum(modifier_VPD, modifier_soilwater) * modifier_age


def calc_canopy_cover(stand_age, LAI, fullCanAge, canpower, k):
    """Canopy cover fraction and light interception.

    Args:
        stand_age: Current stand age (years).
        LAI: Leaf area index (m2/m2).
        fullCanAge: Age at full canopy cover (years).
        canpower: Power controlling canopy closure trajectory (1 = linear).
        k: Radiation extinction coefficient (Beer-Lambert).

    Returns:
        (canopy_cover, light_interception) both in range [0, 1].
    """
    canopy_cover = clip((stand_age / fullCanAge) ** canpower, 0.0, 1.0)
    light_interception = 1 - exp(-1 * k * LAI)
    return canopy_cover, light_interception


def calc_canopy_production(
    solar_rad, days_in_month, light_interception, canopy_cover,
    modifier_physiology, modifier_nutrition, modifier_temperature,
    modifier_frost, alpha, y,
):
    """Determine gross and net biomass production.

    Follows the radiation-use-efficiency (LUE) approach:
        PAR  = solar_rad * days_in_month * molPAR_MJ
        APAR = PAR * light_interception * canopy_cover
        GPP  = APAR * alpha * modifiers
        NPP  = GPP * y

    Args:
        solar_rad: Daily solar radiation (MJ/m2/day).
        days_in_month: Number of days in current month.
        alpha: Canopy quantum efficiency (molC/molPAR).
        y: Assimilate use efficiency (NPP/GPP ratio, ~0.47).

    Returns:
        (PAR, APAR, APARu, GPPmolc, GPPdm, NPP).
    """
    RAD = solar_rad * days_in_month       # MJ/m2
    PAR = RAD * molPAR_MJ                 # mol/m2
    APAR = PAR * light_interception * canopy_cover
    APARu = APAR * modifier_physiology
    alphaC = alpha * modifier_nutrition * modifier_temperature * modifier_frost
    GPPmolc = APARu * alphaC              # mol/m2
    GPPdm = (GPPmolc * gDM_mol) / GPP_UNIT_CONV  # tDM/ha
    NPP = GPPdm * y
    return PAR, APAR, APARu, GPPmolc, GPPdm, NPP


def canopy_production(
    T_av, VPD, ASW, frost_days, stand_age, LAI, solar_rad,
    days_in_month, CounterforShrub, paras, site_paras,
):
    """Main entry point for the Canopy Production module.

    Computes all environmental modifiers and then calculates GPP/NPP.

    Returns:
        Tuple of (PAR, APAR, APARu, GPPmolc, GPPdm, NPP, modifiers,
        LAIShrub, CounterforShrub).
    """
    # Unpack parameters
    T_min = paras.T_min
    T_max = paras.T_max
    T_opt = paras.T_opt
    CoeffCond = paras.CoeffCond
    MaxASW = site_paras.MaxASW
    SWconst = site_paras.SWconst0
    SWpower = site_paras.SWpower0
    FR = site_paras.FR
    fN0 = paras.fN0
    kF = paras.kF
    MaxAge = site_paras.MaxAge
    rAge = paras.rAge
    nAge = paras.nAge
    fullCanAge = paras.fullCanAge
    canpower = paras.canpower
    k = paras.k
    alpha = paras.alpha
    y = paras.y
    CounterforShrub = paras.CounterforShrub
    KL = paras.KL
    Lsx = paras.Lsx

    # Environmental modifiers
    modifier_temperature = calc_modifier_temp(T_av, T_min, T_max, T_opt)
    modifier_VPD = calc_modifier_VPD(VPD, CoeffCond)
    modifier_soilwater = calc_modifier_soilwater(ASW, MaxASW, SWconst, SWpower)
    modifier_nutrition = calc_modifier_soilnutrition(FR, fN0)
    modifier_frost = calc_modifier_frost(frost_days, kF)
    modifier_age = calc_modifier_age(stand_age, MaxAge, rAge, nAge)
    modifier_physiology = calc_physiological_modifier(
        modifier_VPD, modifier_soilwater, modifier_age
    )

    # Light interception and production
    canopy_cover, light_interception = calc_canopy_cover(
        stand_age, LAI, fullCanAge, canpower, k
    )
    PAR, APAR, APARu, GPPmolc, GPPdm, NPP = calc_canopy_production(
        solar_rad, days_in_month, light_interception, canopy_cover,
        modifier_physiology, modifier_nutrition, modifier_temperature,
        modifier_frost, alpha, y,
    )

    modifiers = concatenate([
        modifier_temperature, modifier_VPD, modifier_soilwater,
        modifier_nutrition, modifier_frost, modifier_age,
        modifier_physiology,
    ])

    # Shrub LAI (open-canopy vs closed-canopy estimate)
    LsOpen = LAI * KL
    LsClosed = Lsx * exp(-k * LAI)
    LAIShrub = minimum(LsOpen, LsClosed)

    return PAR, APAR, APARu, GPPmolc, GPPdm, NPP, modifiers, LAIShrub, CounterforShrub
