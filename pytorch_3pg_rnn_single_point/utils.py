from math import exp, sin, cos, sqrt, acos

pi = 3.1415927


class BookKepper(object):
    """a book keeper class, for output calculation results at each step"""

    def __init__(self, fpath):
        super(BookKepper, self).__init__()
        self.fpath = fpath

    def open(self):
        self.handler = open(self.fpath, "w+")

    def shutdown(self):
        if self.handler:
            self.handler.close()

    def write(self, message):
        self.handler.write(message)


def get_VPD(T_min, T_max):
    VPDx = 6.1078 * exp(17.269 * T_max / (237.3 + T_max))
    VPDn = 6.1078 * exp(17.269 * T_min / (237.3 + T_min))
    res = (VPDx - VPDn) / 2
    return res


def get_day_length(lat, month):
    list_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    day_in_year = sum(list_days[: month + 1]) - 15
    s_lat = sin(pi * lat / 180)
    c_lat = cos(pi * lat / 180)
    sin_dec = 0.4 * sin(0.0172 * (day_in_year - 80))
    cosH0 = -sin_dec * s_lat / (c_lat * sqrt(1 - (sin_dec) ** 2))
    if cosH0 > 1:
        res = 0
    elif cosH0 < -1:
        res = 1
    else:
        res = acos(cosH0) / pi
    return 86400 * res


def get_days_in_month(month):
    list_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return list_days[month]


def get_stand_age(lat, InitialYear, InitialMonth, YearPlanted, MonthPlanted, EndAge):
    # This procedure gets the starting month and intial stand age
    # Determine starting month for each year
    if InitialMonth == 0:
        if lat > 0:
            InitialMonth = 0
        else:
            InitialMonth = 6
    if MonthPlanted == 0:
        if lat > 0:
            MonthPlanted = 0
        else:
            MonthPlanted = 6

    # Assign initial stand age
    if InitialYear < YearPlanted:
        InitialYear = YearPlanted + InitialYear

    # TODO
    stand_age = (InitialYear + InitialMonth / 12.0) - (
        YearPlanted + MonthPlanted / 12.0
    )

    # get and check StartAge
    StartAge = int(stand_age)
    if StartAge < 0:
        str_message = (
            """Invalid age limits.
        The starting age %s must be greater than 0!"""
            % StartAge
        )
        raise Exception(str_message)
    elif StartAge > EndAge:
        str_message = """Invalid age limits.
        The starting age %s is greater than the ending age %s.""" % (
            StartAge,
            EndAge,
        )
        raise Exception(str_message)
    return stand_age, StartAge, InitialYear, InitialMonth, MonthPlanted


if __name__ == "__main__":
    print(get_day_length(1, 0))
    print(get_day_length(1, 2))
    print(get_day_length(1, 11))
