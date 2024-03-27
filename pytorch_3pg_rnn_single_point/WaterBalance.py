# %%
"""
Water Balance Module
"""

from const import Qa, Qb
import torch

minimum = torch.min
maximum = torch.max
clip = torch.clamp


def calc_transpiration_PM(Q, VPD, h, gBL, gC):
    """
    Input:
        Q, Double
        VPD, Double
        h, Double
        gBL, Double
        gC, Double
    Output:
        canopy_transpiration, Double
    Descritpion:
        use Penman-Monteith equation for computing canopy transpiration
        in calcuation, the result is kg / (m^2 day),
        which is conmverted to mm/day in the output
    """
    # The following are constants in the PM formula (Landsberg & Gower, 1997)
    e20 = 2.2  # rate of change of saturated VP with T at 20C
    rhoAir = 1.2  # density of air, kg/m3
    lambda_ = 2460000  # latent heat of vapourisation of H2O (J/kg)
    VPDconv = 0.000622  # convert VPD to saturation deficit = 18/29/1000

    netRad = Qa + Qb * (Q * (10**6) / h)  # Q in MJ/m2/day --> W/m2
    defTerm = rhoAir * lambda_ * (VPDconv * VPD) * gBL
    div = 1 + e20 + gBL / gC
    Etransp = (e20 * netRad + defTerm) / div  # in J/m2/s
    canopy_transpiration = Etransp / lambda_ * h  # converted to kg/m2/day
    return canopy_transpiration


def calc_interception(rain, LAI, LAImaxIntcptn, MaxIntcptn):
    eps = 1e-6
    Intcptn = MaxIntcptn * clip(LAI / (LAImaxIntcptn + eps), 0, 1)
    intercepted_water = Intcptn * rain
    return intercepted_water


def calc_soil_water_balance(ASW, rain, loss_water, irrig, MinASW, MaxASW):
    ASW = ASW + rain + (100 * irrig / 12) - loss_water  # Irrig is Ml/ha/year

    monthlyIrrig = maximum(minimum(MinASW - ASW, MinASW), 0)
    ASW = maximum(minimum(ASW, MaxASW), MinASW)

    return ASW, monthlyIrrig


def water_balance(
    solar_rad,
    VPD,
    day_length,
    LAI,
    rain,
    irrig,
    days_in_month,
    ASW,
    CanCond,
    LAIShrub,
    paras,
    site_paras,
):
    BLcond = paras.BLcond

    LAImaxIntcptn = paras.LAImaxIntcptn
    MaxIntcptn = paras.MaxIntcptn

    MinASW = site_paras.MinASW  # [1]
    MaxASW = site_paras.MaxASW  # [0]

    TrShrub = paras.TrShrub

    transp = clip(
        calc_transpiration_PM(solar_rad, VPD, day_length, BLcond, CanCond), 0, None
    )

    transpall = (
        days_in_month * transp * (LAIShrub * TrShrub + LAI) / LAI
    )  # total transpiration
    transp = days_in_month * transp  # tree only transpiration
    transpshrub = clip(transpall - transp, 0, None)  # shrub only transpiration

    intercepted_water = calc_interception(rain, LAI, LAImaxIntcptn, MaxIntcptn)

    loss_water = transp + intercepted_water

    ASW, monthlyIrrig = calc_soil_water_balance(
        ASW, rain, loss_water, irrig, MinASW, MaxASW
    )
    return transpall, transp, transpshrub, loss_water, ASW, monthlyIrrig
