from math import log


"""
Parameters in Canopy Production Module
"""

# Parameters for CanopyProduction/ResponseFunction/Temperature
T_max = 32  # "Critical" biological temperatures: max, min
T_min = 2  # and optimum. Reset if necessary/appropriate
T_opt = 20

# Parameters for CanopyProduction/ResponseFunction/VaporPressureDeficit
CoeffCond = 0.08  # Determines response of canopy conductance to VPD


fN0 = 1  # Value of fN when FR = 0

# Parameters for CanopyProduction/ResponseFunction/Frost
kF = 1  # Number of days production lost per frost days


# Parameters for CanopyProduction/ResponseFunction/Aage
MaxAge = (
    250  # Determines rate of "physiological decline" of forest, move to site_specifi.py
)
nAge = 4  # Parameters in age-modifier
rAge = 0.95  # Relative age to give fAge = 0.5

# Parameters for CanopyProduction/CanopyCoverAndLightInterception
fullCanAge = 20  # Age at full canopy cover
k = 0.5  # Radiation extinction coefficient
canpower = 1  # power term used for describe the trajectory of canopy closure
# 1 means linear trajectory, i.e. original 3PG

# Parameters for CanopyProduction/CanopyProductionCalculation
alpha = 0.0653  # Canopy quantum efficiency
y = 0.47  # Assimilate use efficiency

"""
End of Parameters in Canopy Production Module
"""


"""
Parameters in Biomass Partition Module
"""

# Parameters for calculating canopy conductance
MaxCond = 0.0135  # Maximum canopy conductance (gc, m/s)
LAIgcx = 3.33  # LAI required for maximum canopy conductance
TK2 = 0.244  # added parameter for T modifier
TK3 = 0.0368  # added parameter for T modifier

# Parameters for calculating biomass partitioning coefficients for root, stem and foliage
m0 = 0  # Value of m when FR = 0
# FR                   # same as FR in CanopyProduction/ResponseFunction/SoilNutrition
pRx = 0.45  # maximum root biomass partitioning
pRn = 0.25  # minimum root biomass partitioning
pFS2 = 0.55  # Foliage:stem partitioning ratios for D = 2cm
pFS20 = 0.23  # and D = 20cm

pfsPower = log(pFS20 / pFS2) / log(20 / 2)
pfsConst = pFS2 / (2**pfsPower)

# Parameters for calculating litter fall and root turnover
gammaFx = 0.011261  # Max monthly litterfall rate
gammaF0 = 0.001  # Coefficients in monthly litterfall rate at t=0
tgammaF = 24  # Age at which litterfall rate has median value
Rttover = 0.015  # Root turnover rate per month

# Parameters for d13 calculation
RGcGW = 0.66  # conductance CO2 to water
D13CTissueDif = 1.99  # added parameter for stable isotopes
aFracDiffu = 4.4  # added parameter for stable isotopes
bFracRubi = 27  # added parameter for stable isotopes


"""
End of Parameters in Biomass Partition Module
"""


"""
Parameters in Water Balance Module
"""

BLcond = 0.2  # Canopy boundary layer conductance, assumed constant
LAImaxIntcptn = 0.0  # LAI required for maximum rainfall interception
MaxIntcptn = 0.189  # Max proportion of rainfall intercepted by canopy

"""
End of Parameters in Water Balance Module
"""


"""
Parameters in Stem Mortality Module
"""

SLA0 = 5.633  # specific leaf area at age 0 (m^2/kg)
SLA1 = 5.633  # specific leaf area for mature trees (m^2/kg)
tSLA = 2.5  # stand age (years) for SLA = (SLA0+SLA1)/2
fracBB0 = 0.15  # branch & bark fraction at age 0 (m^2/kg)
fracBB1 = 0.15  # branch & bark fraction for mature trees (m^2/kg)
tBB = 1.5  # stand age (years) for fracBB = (fracBB0+fracBB1)/2

StemConst = 0.058136  # Stem allometric parameters
StemPower = 2.548944  # Stem allometric parameters
wSx1000 = 245  # Max tree stem mass (kg) likely in mature stands of 1000 trees/ha
thinPower = 1.5  # Power in self-thinning law
mF = 0.0  # Leaf mortality fraction
mR = 0.2  # Root mortality fraction
mS = 0.2  # Stem mortality fraction
Density = 0.3805  # basic density (t/m3)

HtC0 = 5.00233  # added parameter for tree height
HtC1 = -8.19365  # added parameter for tree height

"""
End of Parameters in Stem Mortality Module
"""

"""
Start of Parameters for the shrub effect
"""

KL = 0  # ratio of shrub LAI over tree LAI at stand-establishing stage
TrShrub = 0.8  # ratio of the per-LAI transpiration of the shurb to the tree
Lsx = 4  # theoretical maximum LAI that shrub may reach if trees were absent and all resources were abundant
CounterforShrub = 0  # initial value of the indicator for consideration of shrub effect
# 0: shrub growth not limited by light; 1: shrub growth limited by light

"""
End of Parameters for the shrub effect
"""
