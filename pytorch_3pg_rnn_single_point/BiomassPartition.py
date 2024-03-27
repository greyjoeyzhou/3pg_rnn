# %%
"""
Biomass Partitioning Module
"""
import numpy as np

import torch

exp = torch.exp
log = torch.log
clip = torch.clamp


def cacl_canopy_conductance(T_av, LAI, modifier_physiology, TK2, TK3, MaxCond, LAIgcx):
    # calculate canopy conductance from stomatal conductance
    # with added temperature modifier_ Liang Wei
    canopy_conductance = (
        clip(TK2 + TK3 * T_av, 0, 1)
        * MaxCond
        * modifier_physiology
        * clip(LAI / LAIgcx, -np.inf, 1)
    )
    #     if canopy_conductance == 0:
    #         canopy_conductance = 0.0001
    canopy_conductance = clip(canopy_conductance, 0.0001, None)
    return canopy_conductance


def calc_biomass_partition(
    NPP, avDBH, modifier_physiology, m0, FR, pfsConst, pfsPower, pRx, pRn
):
    # calculate partitioning coefficients
    m = m0 + (1 - m0) * FR
    pFS = pfsConst * (avDBH**pfsPower)  # foliage and stem partition
    pR = pRx * pRn / (pRn + (pRx - pRn) * modifier_physiology * m)  # root partition
    pS = (1 - pR) / (1 + pFS)  # stem partition
    pF = 1 - pR - pS  # foliage partition

    # calculate biomass increments
    delWF = NPP * pF  # foilage
    delWR = NPP * pR  # root
    delWS = NPP * pS  # stem
    # print('pRx', pRx)
    # print('pRn', pRn)
    # print('modifier_physiology', modifier_physiology.shape)
    # print('pR', pR.shape)
    return delWF, delWR, delWS


def calc_litter_and_rootturnover(WF, WR, stand_age, gammaFx, gammaF0, tgammaF, Rttover):
    # calculate litterfall & root turnover -
    Littfall = (
        gammaFx
        * gammaF0
        / (
            gammaF0
            + (gammaFx - gammaF0)
            * exp(-12 * log(1 + gammaFx / gammaF0) * stand_age / tgammaF)
        )
    )

    delLitter = Littfall * WF
    delRoots = Rttover * WR

    # print('Littfall', Littfall.shape)
    # print('WF', WF.shape)
    # print('delLitter', delLitter.shape)
    return delLitter, delRoots


def update_endofmonth_biomass(
    WF, WR, WS, TotalLitter, delWF, delWR, delWS, delLitter, delRoots
):
    # Calculate end-of-month biomass
    # print('WS before', WS.shape)
    WF = WF + delWF - delLitter
    WR = WR + delWR - delRoots
    WS = WS + delWS
    TotalW = WF + WR + WS
    # print('delLitter', delLitter.shape)
    # print('TotalLitter before', TotalLitter.shape)
    TotalLitter = TotalLitter + delLitter
    # print('TotalLitter after', TotalLitter.shape)
    # print('WS after', WS.shape)
    return WF, WR, WS, TotalW, TotalLitter


def calc_d13c(
    T_av,
    CaMonthly,
    D13Catm,
    elev,
    GPPmolc,
    days_in_month,
    canopy_conductance,
    RGcGW,
    D13CTissueDif,
    aFracDiffu,
    bFracRubi,
):
    # calculating d13C by Liang Wei

    # Air pressure, kpa
    AirPressure = 101.3 * exp(-1 * elev / 8200)
    # Convert Unit of Atmospheric C, a ppm to part/part
    AtmCa = CaMonthly * 0.000001
    # Canopy conductance for water vapor in mol/m2s, unit conversion
    GwMol = (
        canopy_conductance * 44.6 * (273.15 / (273.15 + T_av)) * (AirPressure / 101.3)
    )
    # Canopy conductance for CO2 in mol/m2s
    GcMol = GwMol * RGcGW

    # GPP per second. Unit: mol/m2 s. GPPmolc divide by 24 hour/day and 3600 s/hr
    GPPmolsec = GPPmolc / (days_in_month * 24 * 3600)

    # Calculating monthly average intercellular CO2 concentration. Ci = Ca - A/g
    InterCi = AtmCa - GPPmolsec / GcMol
    InterCiPPM = InterCi * 1000000

    # Calculating monthly d13C of new photosynthate, = d13Catm- a-(b-a) (ci/ca)
    D13CNewPS = D13Catm - aFracDiffu - (bFracRubi - aFracDiffu) * (InterCi / AtmCa)
    D13CTissue = D13CNewPS + D13CTissueDif
    return D13CTissue, InterCiPPM


def biomass_partion(
    T_av,
    LAI,
    elev,
    CaMonthly,
    D13Catm,
    WF,
    WR,
    WS,
    TotalLitter,
    NPP,
    GPPmolc,
    stand_age,
    days_in_month,
    avDBH,
    modifier_physiology,
    paras,
    site_paras,
):
    # print('TotalLitter initial', TotalLitter.shape)
    TK2 = paras.TK2
    TK3 = paras.TK3
    MaxCond = paras.MaxCond
    LAIgcx = paras.LAIgcx

    m0 = paras.m0
    FR = site_paras.FR  # [4]
    pRx = paras.pRx
    pRn = paras.pRn
    pfsConst = paras.pfsConst
    pfsPower = paras.pfsPower

    gammaFx = paras.gammaFx
    gammaF0 = paras.gammaF0
    tgammaF = paras.tgammaF
    Rttover = paras.Rttover

    RGcGW = paras.RGcGW
    D13CTissueDif = paras.D13CTissueDif
    aFracDiffu = paras.aFracDiffu
    bFracRubi = paras.bFracRubi

    canopy_conductance = cacl_canopy_conductance(
        T_av, LAI, modifier_physiology, TK2, TK3, MaxCond, LAIgcx
    )
    delWF, delWR, delWS = calc_biomass_partition(
        NPP, avDBH, modifier_physiology, m0, FR, pfsConst, pfsPower, pRx, pRn
    )
    delLitter, delRoots = calc_litter_and_rootturnover(
        WF, WR, stand_age, gammaFx, gammaF0, tgammaF, Rttover
    )
    WF, WR, WS, TotalW, TotalLitter = update_endofmonth_biomass(
        WF, WR, WS, TotalLitter, delWF, delWR, delWS, delLitter, delRoots
    )
    D13CTissue, InterCiPPM = calc_d13c(
        T_av,
        CaMonthly,
        D13Catm,
        elev,
        GPPmolc,
        days_in_month,
        canopy_conductance,
        RGcGW,
        D13CTissueDif,
        aFracDiffu,
        bFracRubi,
    )

    return WF, WR, WS, TotalW, TotalLitter, D13CTissue, InterCiPPM, canopy_conductance
