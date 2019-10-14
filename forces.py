from math import sqrt
from numba import jit


@jit(nopython=True)
def f_pp(point1, point2, epsilon, IntRange, radius, ppType='WCA', x_shift=0, y_shift=0):
    # force on point 1 due to point 2.
    # epsilon is the constant for both WCA, and it the spring constant for harmonic.
    x1, y1 = point1
    x2, y2 = point2
    x12 = x1 - (x2 - x_shift)
    y12 = y1 - (y2 - y_shift)
    r_sqrd = x12 ** 2 + y12 ** 2
    f_x, f_y = 0, 0
    if r_sqrd == 0:
        print('division by zero in forces.py')
    elif r_sqrd < IntRange ** 2:
        # shift due to periodic boundary conditions
        if ppType == 'WCA':
            sigma = 2 * radius
            g = -4 * epsilon * (-12 * (sigma ** 2 / r_sqrd) ** 6 +
                                6 * (sigma ** 2 / r_sqrd) ** 3)
            f_x = g * x12 / r_sqrd
            f_y = g * y12 / r_sqrd
        if ppType == 'Harmonic':
            r = sqrt(r_sqrd)
            f_x = epsilon * x12 / r
            f_y = epsilon * y12 / r
    return f_x, f_y


@jit(nopython=True)
def wall_force_linear(y, f_max_N, y_wl, y_wr, ChannelLength_um):
    f = 0
    if y < y_wl:
        f = f_max_N - (f_max_N / y_wl) * y
    if y > y_wr:
        f = -f_max_N / (y_wr - ChannelLength_um) * (y_wr - y)
    return f

@jit(nopython=True)
def wall_force_biharm(y, f_max_N, y_wl, y_wr, ChannelLength_um):
    f = 0
    if y < y_wl:
        f = -f_max_N/(y_wl**3)*(y-y_wl)**3
    if y > y_wr:
        f = f_max_N/(y_wr-ChannelLength_um)**3*(y-y_wr)**3
    return f

@jit(nopython=True)
def exp_wall_force(y, ChannelLength_um):
    ## for 20190114-S25-dilute, 20190124_S20\DEP20X1p5_S20_2TMA_0H2O2_1p6pp_004_EnergyForceProfiles_MetaData.txt
    # a = 1.60064484 * 10 ** (-6)
    # n = -6
    # x1 = -24.8416038
    # x2 = ChannelLength_um - x1
    ## for 20190212_dblLineAsym2, 20190214_dblLineAsym\DEP20X1p5X_S25_2TMA_0H2O2_0p64pp_flippedDevice_007_EnergyForceProfiles.csv WITH n=-4
    # a = 1*10**(-9)
    # n = -4
    # x1 = -21
    # x2 = ChannelLength_um - x1
    ## for 20190212_dblLineAsym2, 20190214_dblLineAsym\DEP20X1p5X_S25_2TMA_0H2O2_0p64pp_flippedDevice_008_EnergyForceProfiles.csv WITH n=-6
    a = 3*10**(-7)
    n = -6
    x1 = -13.5
    x2 = ChannelLength_um - x1
    ## for 20190212_dblLineAsym2, 20190214_dblLineAsym\DEP20X1p5X_S25_2TMA_0H2O2_0p64pp_flippedDevice_007_EnergyForceProfiles.csv"
    # a = 1.4 * 10 ** (-6)
    # n = -6
    # x1 = -24.8416038
    # x2 = ChannelLength_um - x1

    f= a * (y - x1) ** n - a * (y - x2) ** n
    return f
