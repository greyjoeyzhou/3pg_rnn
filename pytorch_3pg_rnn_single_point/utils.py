"""Calendar, astronomical, and stand-age utility functions for the 3-PG model.

Provides day-length calculation (solar geometry), days-in-month lookup,
vapour-pressure-deficit estimation, and initial stand-age validation.
"""

from math import exp, sin, cos, sqrt, acos, pi


class BookKeeper:
    """Simple file writer for saving model output at each time step."""

    def __init__(self, fpath):
        self.fpath = fpath
        self.handler = None

    def open(self):
        self.handler = open(self.fpath, "w+")

    def shutdown(self):
        if self.handler:
            self.handler.close()

    def write(self, message):
        self.handler.write(message)


# Month-length lookup (non-leap year)
_DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def get_VPD(T_min, T_max):
    """Estimate mean daily vapour-pressure deficit from min/max temperature.

    Uses the Magnus formula for saturation vapour pressure.

    Args:
        T_min: Minimum daily temperature (deg C).
        T_max: Maximum daily temperature (deg C).

    Returns:
        Mean VPD (kPa).
    """
    VPDx = 6.1078 * exp(17.269 * T_max / (237.3 + T_max))
    VPDn = 6.1078 * exp(17.269 * T_min / (237.3 + T_min))
    return (VPDx - VPDn) / 2


def get_day_length(lat, month):
    """Calculate day length from latitude and month.

    Uses solar declination geometry to compute the fraction of the day
    with sunlight, returned as seconds of daylight.

    Args:
        lat: Latitude in decimal degrees (positive = N, negative = S).
        month: Zero-indexed month (0 = January, 11 = December).

    Returns:
        Day length in seconds.
    """
    day_in_year = sum(_DAYS_IN_MONTH[: month + 1]) - 15
    s_lat = sin(pi * lat / 180)
    c_lat = cos(pi * lat / 180)
    sin_dec = 0.4 * sin(0.0172 * (day_in_year - 80))
    cosH0 = -sin_dec * s_lat / (c_lat * sqrt(1 - sin_dec ** 2))

    if cosH0 > 1:
        return 0.0
    elif cosH0 < -1:
        return 86400.0
    else:
        return 86400.0 * acos(cosH0) / pi


def get_days_in_month(month):
    """Return the number of days in a zero-indexed month (non-leap year).

    Args:
        month: Zero-indexed month (0 = January, 11 = December).
    """
    return _DAYS_IN_MONTH[month]


def get_stand_age(lat, InitialYear, InitialMonth, YearPlanted, MonthPlanted, EndAge):
    """Compute the initial stand age and validate time-range parameters.

    Adjusts starting months for hemisphere and checks that StartAge is
    between 0 and EndAge (inclusive).

    Args:
        lat: Latitude (degrees).
        InitialYear: Calendar year of the first observation.
        InitialMonth: Zero-indexed starting month (0 = auto-detect).
        YearPlanted: Calendar year the stand was planted.
        MonthPlanted: Zero-indexed planting month (0 = auto-detect).
        EndAge: Maximum stand age to simulate (years).

    Returns:
        Tuple of (stand_age, StartAge, InitialYear, InitialMonth, MonthPlanted).

    Raises:
        Exception: If starting age is negative or exceeds EndAge.
    """
    if InitialMonth == 0:
        InitialMonth = 0 if lat > 0 else 6
    if MonthPlanted == 0:
        MonthPlanted = 0 if lat > 0 else 6

    if InitialYear < YearPlanted:
        InitialYear = YearPlanted + InitialYear

    stand_age = (InitialYear + InitialMonth / 12.0) - (
        YearPlanted + MonthPlanted / 12.0
    )

    StartAge = int(stand_age)
    if StartAge < 0:
        raise Exception(
            f"Invalid age limits. The starting age {StartAge} must be >= 0."
        )
    if StartAge > EndAge:
        raise Exception(
            f"Invalid age limits. The starting age {StartAge} "
            f"exceeds the ending age {EndAge}."
        )
    return stand_age, StartAge, InitialYear, InitialMonth, MonthPlanted


if __name__ == "__main__":
    print(get_day_length(1, 0))
    print(get_day_length(1, 2))
    print(get_day_length(1, 11))
