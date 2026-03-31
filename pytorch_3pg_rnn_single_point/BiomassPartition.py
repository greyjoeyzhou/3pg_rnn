"""Biomass Partitioning Module for the 3-PG model.

Allocates net primary production (NPP) to foliage, roots, and stems,
computes litterfall and root turnover, and calculates stable carbon
isotope discrimination (delta-13C).

References:
    Landsberg & Waring (1997), Forest Ecology and Management 95, 209-228.
    Wei et al. — delta-13C tissue calculation extensions.
"""

import numpy as np

from const import (
    STANDARD_PRESSURE, PRESSURE_SCALE_HEIGHT,
    MOL_AIR_VOLUME, KELVIN_OFFSET,
)
from torch_compat import exp, log, clip


def calc_canopy_conductance(T_av, LAI, modifier_physiology, TK2, TK3, MaxCond, LAIgcx):
    """Canopy conductance for water vapour (m/s).

    Scales maximum stomatal conductance by temperature, LAI, and
    physiological modifiers.

    Args:
        T_av: Mean monthly temperature (deg C).
        LAI: Leaf area index (m2/m2).
        modifier_physiology: Combined physiological modifier (0-1).
        TK2, TK3: Temperature modifier coefficients.
        MaxCond: Maximum canopy conductance (m/s).
        LAIgcx: LAI at which conductance reaches MaxCond.

    Returns:
        Canopy conductance (m/s), floored at 0.0001.
    """
    conductance = (
        clip(TK2 + TK3 * T_av, 0, 1)
        * MaxCond
        * modifier_physiology
        * clip(LAI / LAIgcx, -np.inf, 1)
    )
    return clip(conductance, 0.0001, None)


def calc_biomass_partition(NPP, avDBH, modifier_physiology, m0, FR, pfsConst, pfsPower, pRx, pRn):
    """Partition NPP into foliage, root, and stem increments.

    Allocation fractions depend on current DBH (via the foliage:stem ratio)
    and moisture/age stress (via the physiological modifier).

    Args:
        NPP: Net primary production (tDM/ha/month).
        avDBH: Average diameter at breast height (cm).
        modifier_physiology: Combined physiological modifier (0-1).
        m0: Soil-nutrition multiplier floor.
        FR: Fertility rating (0-1).
        pfsConst, pfsPower: Foliage:stem allometric coefficients.
        pRx: Maximum root partition fraction.
        pRn: Minimum root partition fraction.

    Returns:
        (delWF, delWR, delWS) — biomass increments (tDM/ha/month).
    """
    m = m0 + (1 - m0) * FR
    pFS = pfsConst * (avDBH ** pfsPower)
    pR = pRx * pRn / (pRn + (pRx - pRn) * modifier_physiology * m)
    pS = (1 - pR) / (1 + pFS)
    pF = 1 - pR - pS

    delWF = NPP * pF
    delWR = NPP * pR
    delWS = NPP * pS
    return delWF, delWR, delWS


def calc_litter_and_rootturnover(WF, WR, stand_age, gammaFx, gammaF0, tgammaF, Rttover):
    """Monthly litterfall and root turnover.

    Litterfall rate increases with age from gammaF0 towards gammaFx,
    reaching its median at age = tgammaF.

    Args:
        WF: Foliage biomass (tDM/ha).
        WR: Root biomass (tDM/ha).
        stand_age: Stand age (years).
        gammaFx: Maximum monthly litterfall rate.
        gammaF0: Initial monthly litterfall rate.
        tgammaF: Age at which rate reaches median (years).
        Rttover: Monthly root turnover fraction.

    Returns:
        (delLitter, delRoots) — biomass losses (tDM/ha/month).
    """
    Littfall = (
        gammaFx * gammaF0
        / (
            gammaF0
            + (gammaFx - gammaF0)
            * exp(-12 * log(1 + gammaFx / gammaF0) * stand_age / tgammaF)
        )
    )
    delLitter = Littfall * WF
    delRoots = Rttover * WR
    return delLitter, delRoots


def update_endofmonth_biomass(WF, WR, WS, TotalLitter, delWF, delWR, delWS, delLitter, delRoots):
    """Update biomass pools at end of month.

    Returns:
        (WF, WR, WS, TotalW, TotalLitter).
    """
    WF = WF + delWF - delLitter
    WR = WR + delWR - delRoots
    WS = WS + delWS
    TotalW = WF + WR + WS
    TotalLitter = TotalLitter + delLitter
    return WF, WR, WS, TotalW, TotalLitter


def calc_d13c(
    T_av, CaMonthly, D13Catm, elev, GPPmolc, days_in_month,
    canopy_conductance, RGcGW, D13CTissueDif, aFracDiffu, bFracRubi,
):
    """Compute delta-13C of new photosynthate and tree-ring tissue.

    Based on the Farquhar model of carbon isotope discrimination:
        delta-13C = delta-13C_atm - a - (b - a) * (Ci / Ca)

    where Ci is intercellular CO2 derived from GPP and canopy conductance.

    Args:
        T_av: Mean temperature (deg C).
        CaMonthly: Atmospheric CO2 concentration (ppm).
        D13Catm: delta-13C of atmospheric CO2 (per mille).
        elev: Site elevation (m).
        GPPmolc: Gross primary production (mol C / m2 / month).
        days_in_month: Days in the current month.
        canopy_conductance: Canopy conductance for water vapour (m/s).
        RGcGW: Ratio of CO2 conductance to H2O conductance (~0.66).
        D13CTissueDif: Tissue-photosynthate fractionation offset (per mille).
        aFracDiffu: Fractionation during diffusion through stomata (per mille).
        bFracRubi: Fractionation by Rubisco (per mille).

    Returns:
        (D13CTissue, InterCiPPM) — tissue delta-13C and intercellular CO2 (ppm).
    """
    AirPressure = STANDARD_PRESSURE * exp(-1 * elev / PRESSURE_SCALE_HEIGHT)
    AtmCa = CaMonthly * 1e-6  # ppm -> mol fraction

    # Canopy conductance for water vapour -> mol/m2/s
    GwMol = (
        canopy_conductance
        * MOL_AIR_VOLUME
        * (KELVIN_OFFSET / (KELVIN_OFFSET + T_av))
        * (AirPressure / STANDARD_PRESSURE)
    )
    GcMol = GwMol * RGcGW  # CO2 conductance

    # GPP per second (mol/m2/s)
    GPPmolsec = GPPmolc / (days_in_month * 24 * 3600)

    # Intercellular CO2: Ci = Ca - A/g
    InterCi = AtmCa - GPPmolsec / GcMol
    InterCiPPM = InterCi * 1e6

    # delta-13C of new photosynthate
    D13CNewPS = D13Catm - aFracDiffu - (bFracRubi - aFracDiffu) * (InterCi / AtmCa)
    D13CTissue = D13CNewPS + D13CTissueDif
    return D13CTissue, InterCiPPM


def biomass_partition(
    T_av, LAI, elev, CaMonthly, D13Catm,
    WF, WR, WS, TotalLitter,
    NPP, GPPmolc, stand_age, days_in_month, avDBH,
    modifier_physiology, paras, site_paras,
):
    """Main entry point for the Biomass Partitioning module.

    Computes canopy conductance, allocates NPP, updates biomass pools,
    and calculates delta-13C.

    Returns:
        (WF, WR, WS, TotalW, TotalLitter, D13CTissue, InterCiPPM,
         canopy_conductance).
    """
    canopy_conductance = calc_canopy_conductance(
        T_av, LAI, modifier_physiology,
        paras.TK2, paras.TK3, paras.MaxCond, paras.LAIgcx,
    )

    delWF, delWR, delWS = calc_biomass_partition(
        NPP, avDBH, modifier_physiology,
        paras.m0, site_paras.FR, paras.pfsConst, paras.pfsPower,
        paras.pRx, paras.pRn,
    )

    delLitter, delRoots = calc_litter_and_rootturnover(
        WF, WR, stand_age,
        paras.gammaFx, paras.gammaF0, paras.tgammaF, paras.Rttover,
    )

    WF, WR, WS, TotalW, TotalLitter = update_endofmonth_biomass(
        WF, WR, WS, TotalLitter, delWF, delWR, delWS, delLitter, delRoots,
    )

    D13CTissue, InterCiPPM = calc_d13c(
        T_av, CaMonthly, D13Catm, elev, GPPmolc, days_in_month,
        canopy_conductance,
        paras.RGcGW, paras.D13CTissueDif, paras.aFracDiffu, paras.bFracRubi,
    )

    return WF, WR, WS, TotalW, TotalLitter, D13CTissue, InterCiPPM, canopy_conductance
