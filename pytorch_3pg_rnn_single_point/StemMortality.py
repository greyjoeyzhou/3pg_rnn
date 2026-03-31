"""Stem Mortality Module for the 3-PG model.

Applies the self-thinning rule to update stem density, recalculates
age-dependent factors (SLA, branch/bark fraction), and derives stand
metrics (LAI, DBH, basal area, height, volume, MAI).

References:
    Landsberg & Waring (1997), Forest Ecology and Management 95, 209-228.
    Wykoff (1982) — height-diameter equation.
"""

from math import pi

import torch

from const import KG_PER_T, SLA_LAI_CONV
from torch_compat import exp, log, cast, where


def getMortality(oldN, oldW, mS, wSx1000, thinPower):
    """Number of stems to remove to satisfy the self-thinning rule.

    Solves: N_new = 1000 * (wSx1000 * N / W / 1000) ^ (1 / thinPower)
    and returns oldN - N_new, rounded to the nearest integer.

    Args:
        oldN: Current stem density (stems/ha).
        oldW: Current stem biomass (tDM/ha).
        mS: Stem mortality fraction.
        wSx1000: Maximum stem mass at 1000 stems/ha (kg).
        thinPower: Self-thinning exponent.

    Returns:
        Number of stems to remove (rounded).
    """
    res = oldN - 1000 * (wSx1000 * oldN / oldW / 1000) ** (1 / thinPower)
    return torch.round(res)


def calc_mortality(WF, WR, WS, StemNo, delStemNo, wSx1000, thinPower, mF, mR, mS):
    """Apply self-thinning mortality and update biomass pools.

    Mortality only occurs when average stem mass exceeds the maximum
    allowed by the self-thinning law for the current stem density.

    Args:
        WF, WR, WS: Foliage, root, stem biomass (tDM/ha).
        StemNo: Current stem density (stems/ha).
        delStemNo: Cumulative stems removed.
        wSx1000: Max stem mass at 1000 stems/ha (kg).
        thinPower: Self-thinning exponent.
        mF, mR, mS: Mortality fractions for foliage, roots, stems.

    Returns:
        (WF, WR, WS, AvStemMass, StemNo, delStemNo).
    """
    wSmax = wSx1000 * (1000 / StemNo) ** thinPower
    AvStemMass = WS * KG_PER_T / StemNo

    # Only thin when average stem mass exceeds the self-thinning limit
    delStems = cast(wSmax < AvStemMass, "float32") * getMortality(
        StemNo, WS, mS, wSx1000, thinPower
    )

    WF = WF - mF * delStems * (WF / StemNo)
    WR = WR - mR * delStems * (WR / StemNo)
    WS = WS - mS * delStems * (WS / StemNo)

    StemNo = StemNo - delStems
    AvStemMass = WS * KG_PER_T / StemNo
    delStemNo = delStemNo + delStems
    return WF, WR, WS, AvStemMass, StemNo, delStemNo


def calc_factors_age(stand_age, SLA0, SLA1, tSLA, fracBB0, fracBB1, tBB):
    """Age-dependent specific leaf area and branch/bark fraction.

    Both follow a decay from initial (age-0) to mature values, with
    half-life parameters tSLA and tBB respectively.

    Args:
        stand_age: Current stand age (years).
        SLA0: SLA at age 0 (m2/kg).
        SLA1: SLA at maturity (m2/kg).
        tSLA: Age for SLA midpoint (years).
        fracBB0: Branch+bark fraction at age 0.
        fracBB1: Branch+bark fraction at maturity.
        tBB: Age for fracBB midpoint (years).

    Returns:
        (SLA, fracBB).
    """
    SLA = SLA1 + (SLA0 - SLA1) * exp(-log(2.0) * (stand_age / tSLA) ** 2)
    fracBB = fracBB1 + (fracBB0 - fracBB1) * exp(-log(2.0) * (stand_age / tBB))
    return SLA, fracBB


def update_stands(
    stand_age, WF, WS, AvStemMass, StemNo,
    SLA, fracBB, StemConst, StemPower, Density, HtC0, HtC1,
):
    """Derive stand-level metrics from biomass and stem density.

    Args:
        stand_age: Stand age (years).
        WF: Foliage biomass (tDM/ha).
        WS: Stem biomass (tDM/ha).
        AvStemMass: Average stem mass (kg/tree).
        StemNo: Stem density (stems/ha).
        SLA: Specific leaf area (m2/kg).
        fracBB: Branch and bark fraction.
        StemConst, StemPower: DBH allometric coefficients.
        Density: Basic wood density (t/m3).
        HtC0, HtC1: Height equation coefficients (Wykoff 1982, imperial units).

    Returns:
        (LAI, MAI, avDBH, BasArea, Height, StandVol).
    """
    LAI = WF * SLA * SLA_LAI_CONV
    avDBH = (AvStemMass / StemConst) ** (1 / StemPower)
    BasArea = (((avDBH / 200) ** 2) * pi) * StemNo
    StandVol = WS * (1 - fracBB) / Density
    MAI = where(stand_age > 0, StandVol / stand_age, 0.0)

    # Wykoff (1982): DBH in inches, height in feet -> convert to metric
    Height = (exp(HtC0 + HtC1 / (avDBH / 2.54 + 1)) + 4.5) * 0.3048

    return LAI, MAI, avDBH, BasArea, Height, StandVol


def stem_mortality(
    WF, WR, WS, StemNo, delStemNo, stand_age, paras,
    doThinning=None, doDefoliation=None,
):
    """Main entry point for the Stem Mortality module.

    Advances stand age by one month, applies self-thinning mortality,
    recalculates age-dependent factors, and updates stand metrics.

    Returns:
        (stand_age, LAI, MAI, avDBH, BasArea, Height, StemNo,
         delStemNo, StandVol, WF, WR, WS).
    """
    if doThinning is not None:
        doThinning()
    if doDefoliation is not None:
        doDefoliation()

    stand_age = stand_age + 1.0 / 12

    WF, WR, WS, AvStemMass, StemNo, delStemNo = calc_mortality(
        WF, WR, WS, StemNo, delStemNo,
        paras.wSx1000, paras.thinPower, paras.mF, paras.mR, paras.mS,
    )

    SLA, fracBB = calc_factors_age(
        stand_age,
        paras.SLA0, paras.SLA1, paras.tSLA,
        paras.fracBB0, paras.fracBB1, paras.tBB,
    )

    LAI, MAI, avDBH, BasArea, Height, StandVol = update_stands(
        stand_age, WF, WS, AvStemMass, StemNo, SLA, fracBB,
        paras.StemConst, paras.StemPower, paras.Density,
        paras.HtC0, paras.HtC1,
    )

    return (
        stand_age, LAI, MAI, avDBH, BasArea, Height,
        StemNo, delStemNo, StandVol, WF, WR, WS,
    )
