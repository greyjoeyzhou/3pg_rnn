# %%
"""
Stem Mortality Module
"""


from numpy import pi
import torch

exp = torch.exp


def log(x):
    return torch.log(torch.tensor(x))


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


round = torch.round


def where(condition, input, other):
    return torch.where(condition, input, other)


"""
    # Perform any thinning or defoliation events for this time period
    if thinEventNo <= nThinning:
        doThinning(thinEventNo, Thinning)
    if defoltnEventNo <= nDefoliation:
        doDefoliation(defoltnEventNo, Defoliation)
"""


def getMortality(oldN, oldW, mS, wSx1000, thinPower):
    """
    Input:
        oldN, Double
        oldW, Double
    Output:
        mortality rate, Double
    Description:
        This function determines the number of stems to remove to ensure the
        self-thinning rule is satisfied. It applies the Newton-Rhapson method
        to solve for N to an accuracy of 1 stem or less. To change this,
        change the value of "accuracy".

        This was the old mortality function:
          getMortality = oldN - 1000 * (wSx1000 * oldN / oldW / 1000) ^ (1 / thinPower)
        which has been superceded by the following ...
    """

    res = oldN - 1000 * (wSx1000 * oldN / oldW / 1000) ** (1 / thinPower)
    return round(res)


def calc_mortality(WF, WR, WS, StemNo, delStemNo, wSx1000, thinPower, mF, mR, mS):
    # Calculate mortality
    wSmax = wSx1000 * (1000 / StemNo) ** thinPower
    AvStemMass = WS * 1000 / StemNo
    delStems = cast(wSmax < AvStemMass, "float32") * getMortality(
        StemNo, WS, mS, wSx1000, thinPower
    )

    WF = WF - mF * delStems * (WF / StemNo)
    WR = WR - mR * delStems * (WR / StemNo)
    WS = WS - mS * delStems * (WS / StemNo)

    StemNo = StemNo - delStems
    AvStemMass = WS * 1000 / StemNo
    delStemNo = delStemNo + delStems
    return WF, WR, WS, AvStemMass, StemNo, delStemNo


def calc_factors_age(stand_age, SLA0, SLA1, tSLA, fracBB0, fracBB1, tBB):
    # update age-dependent factors
    SLA = SLA1 + (SLA0 - SLA1) * exp(-log(2.0) * (stand_age / tSLA) ** 2)
    fracBB = fracBB1 + (fracBB0 - fracBB1) * exp(-log(2.0) * (stand_age / tBB))
    return SLA, fracBB


def update_stands(
    stand_age,
    WF,
    WS,
    AvStemMass,
    StemNo,
    SLA,
    fracBB,
    StemConst,
    StemPower,
    Density,
    HtC0,
    HtC1,
):
    # update stsand characteristics
    LAI = WF * SLA * 0.1
    avDBH = (AvStemMass / StemConst) ** (1 / StemPower)
    BasArea = (((avDBH / 200) ** 2) * pi) * StemNo
    StandVol = WS * (1 - fracBB) / Density
    MAI = where(stand_age > 0, StandVol / stand_age, 0.0)

    # Height equation (Wykoff 1982) is in English unit,
    # DBH is first convert to inch.
    # Finally Ht is convert form feet to meters._Liang
    Height = (exp(HtC0 + HtC1 / (avDBH / 2.54 + 1)) + 4.5) * 0.3048

    return LAI, MAI, avDBH, BasArea, Height, StandVol


def stem_mortality(
    WF, WR, WS, StemNo, delStemNo, stand_age, paras, doThinning=None, doDefoliation=None
):

    wSx1000 = paras.wSx1000
    thinPower = paras.thinPower
    mF = paras.mF
    mR = paras.mR
    mS = paras.mS

    SLA0 = paras.SLA0
    SLA1 = paras.SLA1
    tSLA = paras.tSLA
    fracBB0 = paras.fracBB0
    fracBB1 = paras.fracBB1
    tBB = paras.tBB

    StemConst = paras.StemConst
    StemPower = paras.StemPower
    Density = paras.Density
    HtC0 = paras.HtC0
    HtC1 = paras.HtC1

    if doThinning is not None:
        doThinning()
    if doDefoliation is not None:
        doDefoliation()

    stand_age = stand_age + 1.0 / 12

    WF, WR, WS, AvStemMass, StemNo, delStemNo = calc_mortality(
        WF, WR, WS, StemNo, delStemNo, wSx1000, thinPower, mF, mR, mS
    )
    SLA, fracBB = calc_factors_age(stand_age, SLA0, SLA1, tSLA, fracBB0, fracBB1, tBB)
    LAI, MAI, avDBH, BasArea, Height, StandVol = update_stands(
        stand_age,
        WF,
        WS,
        AvStemMass,
        StemNo,
        SLA,
        fracBB,
        StemConst,
        StemPower,
        Density,
        HtC0,
        HtC1,
    )
    return (
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
    )
