"""Water Balance Module for the 3-PG model.

Computes transpiration (Penman-Monteith), rainfall interception, and
monthly soil-water balance.

References:
    Landsberg & Gower (1997). Applications of Physiological Ecology to
    Forest Management. Academic Press.
"""

from const import Qa, Qb, e20, rhoAir, LAMBDA, VPDconv
from torch_compat import clip, minimum, maximum


def calc_transpiration_PM(Q, VPD, h, gBL, gC):
    """Canopy transpiration via the Penman-Monteith equation.

    Args:
        Q: Daily solar radiation (MJ/m2/day).
        VPD: Vapour pressure deficit (kPa).
        h: Day length (seconds).
        gBL: Boundary-layer conductance (m/s).
        gC: Canopy conductance (m/s).

    Returns:
        Canopy transpiration rate (mm/day).
    """
    netRad = Qa + Qb * (Q * 1e6 / h)              # W/m2
    defTerm = rhoAir * LAMBDA * (VPDconv * VPD) * gBL
    div = 1 + e20 + gBL / gC
    Etransp = (e20 * netRad + defTerm) / div       # J/m2/s
    canopy_transpiration = Etransp / LAMBDA * h     # kg/m2/day ≈ mm/day
    return canopy_transpiration


def calc_interception(rain, LAI, LAImaxIntcptn, MaxIntcptn):
    """Rainfall interception by the canopy (mm).

    Interception fraction scales linearly with LAI up to LAImaxIntcptn.

    Args:
        rain: Monthly rainfall (mm).
        LAI: Leaf area index (m2/m2).
        LAImaxIntcptn: LAI at which interception saturates.
        MaxIntcptn: Maximum interception fraction (0-1).
    """
    eps = 1e-6
    Intcptn = MaxIntcptn * clip(LAI / (LAImaxIntcptn + eps), 0, 1)
    return Intcptn * rain


def calc_soil_water_balance(ASW, rain, loss_water, irrig, MinASW, MaxASW):
    """Update available soil water for the month.

    ASW is bounded by [MinASW, MaxASW] after adding inputs and
    subtracting losses.

    Args:
        ASW: Available soil water at start of month (mm).
        rain: Monthly rainfall (mm).
        loss_water: Transpiration + interception losses (mm).
        irrig: Annual irrigation (ML/ha/yr), distributed evenly.
        MinASW, MaxASW: Soil water bounds (mm).

    Returns:
        (ASW, monthlyIrrig).
    """
    ASW = ASW + rain + (100 * irrig / 12) - loss_water

    monthlyIrrig = maximum(minimum(MinASW - ASW, MinASW), 0)
    ASW = maximum(minimum(ASW, MaxASW), MinASW)

    return ASW, monthlyIrrig


def water_balance(
    solar_rad, VPD, day_length, LAI, rain, irrig, days_in_month,
    ASW, CanCond, LAIShrub, paras, site_paras,
):
    """Main entry point for the Water Balance module.

    Computes tree and shrub transpiration, rainfall interception, and
    updates available soil water.

    Returns:
        (transpall, transp, transpshrub, loss_water, ASW, monthlyIrrig).
    """
    BLcond = paras.BLcond
    LAImaxIntcptn = paras.LAImaxIntcptn
    MaxIntcptn = paras.MaxIntcptn
    MinASW = site_paras.MinASW
    MaxASW = site_paras.MaxASW
    TrShrub = paras.TrShrub

    transp = clip(
        calc_transpiration_PM(solar_rad, VPD, day_length, BLcond, CanCond),
        0, None,
    )

    transpall = days_in_month * transp * (LAIShrub * TrShrub + LAI) / LAI
    transp = days_in_month * transp
    transpshrub = clip(transpall - transp, 0, None)

    intercepted_water = calc_interception(rain, LAI, LAImaxIntcptn, MaxIntcptn)
    loss_water = transp + intercepted_water

    ASW, monthlyIrrig = calc_soil_water_balance(
        ASW, rain, loss_water, irrig, MinASW, MaxASW,
    )
    return transpall, transp, transpshrub, loss_water, ASW, monthlyIrrig
