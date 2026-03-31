"""Physical constants, conversion factors, and named indices for the 3-PG model.

References:
    Landsberg & Waring (1997). A generalised model of forest productivity
    using simplified concepts of radiation-use efficiency, carbon balance
    and partitioning. Forest Ecology and Management, 95(3), 209-228.
"""

# ---------------------------------------------------------------------------
# Unit conversion factors
# ---------------------------------------------------------------------------
gDM_mol = 24       # grams dry matter per mol CO2
molPAR_MJ = 2.3    # mol PAR per MJ solar radiation

# ---------------------------------------------------------------------------
# Radiation
# ---------------------------------------------------------------------------
Qa = -90            # Intercept of net vs. solar radiation relationship (W/m2)
Qb = 0.8            # Slope of net vs. solar radiation relationship

# ---------------------------------------------------------------------------
# Penman-Monteith constants
# ---------------------------------------------------------------------------
e20 = 2.2           # Rate of change of saturated VP with T at 20C (mbar/C)
rhoAir = 1.2        # Density of air (kg/m3)
LAMBDA = 2460000    # Latent heat of vapourisation of H2O (J/kg)
VPDconv = 0.000622  # Convert VPD to saturation deficit (= 18/29/1000)

# ---------------------------------------------------------------------------
# Atmospheric / gas-exchange constants
# ---------------------------------------------------------------------------
STANDARD_PRESSURE = 101.3       # Standard atmospheric pressure (kPa)
PRESSURE_SCALE_HEIGHT = 8200    # Atmospheric pressure scale height (m)
MOL_AIR_VOLUME = 44.6           # Molar volume of air at STP (mol/m3)
KELVIN_OFFSET = 273.15          # Celsius to Kelvin offset

# ---------------------------------------------------------------------------
# Biomass / unit conversion helpers
# ---------------------------------------------------------------------------
KG_PER_T = 1000     # kg per tonne of dry matter
SLA_LAI_CONV = 0.1  # SLA (m2/kg) * WF (tDM/ha) * 0.1 -> LAI (m2/m2)
GPP_UNIT_CONV = 100  # Convert gDM/m2 to tDM/ha

# ---------------------------------------------------------------------------
# Missing-data sentinel
# ---------------------------------------------------------------------------
NODATA = -9999.0    # Sentinel value for missing observations


# ---------------------------------------------------------------------------
# Named indices
# ---------------------------------------------------------------------------
class StateIndex:
    """Named indices for the 13-element model state vector."""
    STAND_VOL = 0
    LAI = 1
    ASW = 2
    STEM_NO = 3
    PAR = 4
    STAND_AGE = 5
    WF = 6
    WR = 7
    WS = 8
    TOTAL_LITTER = 9
    AV_DBH = 10
    DEL_STEM_NO = 11
    D13C_TISSUE = 12
    COUNT = 13

    NAMES = [
        "StandVol", "LAI", "ASW", "StemNo", "PAR", "stand_age",
        "WF", "WR", "WS", "TotalLitter", "avDBH", "delStemNo",
        "D13CTissue",
    ]


class InputIndex:
    """Named indices for the 19-element input vector.

    Channels 0-1 are unused placeholders retained for compatibility.
    """
    T_AV = 2
    VPD = 3
    RAIN = 4
    SOLAR_RAD = 5
    RAIN_DAYS = 6
    FROST_DAYS = 7
    CA_MONTHLY = 8
    D13C_ATM = 9
    DAY_LENGTH = 10
    DAYS_IN_MONTH = 11
    # Site parameters packed into the input tensor:
    MAX_ASW = 12
    MIN_ASW = 13
    SW_CONST = 14
    SW_POWER = 15
    FR = 16
    MAX_AGE = 17
    ELEV = 18
    COUNT = 19
