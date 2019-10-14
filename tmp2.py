from math import sqrt, floor
import numpy as np
from forces import f_pp

kT = 4.11 * 10 ** (-21)
D_T = 0.1
dt = 0.001
D_R = 0.1
ppType = 'WCA'
epsilon = 10 ** (-18)
drag_N_sPum = kT / D_T * 10 ** 6

radius = 1;
BoxX = 200
IntRange = 2 * radius * 2 ** (1 / 6)
BoxY = 100
NumberOfParticles = 2

coordsForRunning = np.array([[0, 199.5, 50, 0, 0.5, 50], [0, 0, 0, 0, 0, 0]])

sigma = 2 * radius
Lcx = int(floor(BoxX / IntRange))  # Number of x cells, # BoxX is Region[0] in the original script;
Lcy = int(floor(BoxY / IntRange))  # Number of y cells, BoxY is Region[1] in the original script;
IntRange_cx = BoxX / Lcx  # edge in um of x cell. rc[0] in original script
IntRange_cy = BoxY / Lcy  # edge in um of y cell. rc[1] in original script
lscl = np.zeros(NumberOfParticles, np.float_)
cell_array_head = np.zeros((Lcx, Lcy),
                           np.float_)  # array with all cells. Will point to the first particle in each cell.
# noise: initial noise for first 10000 steps
x_noises_um = sqrt(2 * D_T * dt) * np.random.randn(10000, NumberOfParticles) + 0
y_noises_um = sqrt(2 * D_T * dt) * np.random.randn(10000, NumberOfParticles) + 0
p_noises = sqrt(2 * D_R * dt) * np.random.randn(10000, NumberOfParticles) + 0
jo = 0
jn = 1
i = 1;
# Run over timesteps
cell_array_head[:, :] = np.nan
lscl[:] = np.nan
for particle in range(NumberOfParticles):
    # Build linked cell list for timestep i, to be used for inter-particle forces
    # http://cacs.usc.edu/education/cs596/01-1LinkedListCell.pdf
    X_cell_ind = int(floor(coordsForRunning[
                               jo, particle * 3 + 1] / IntRange_cx))  # find x coordinate of the cell that contains the particle
    Y_cell_ind = int(floor(coordsForRunning[
                               jo, particle * 3 + 2] / IntRange_cy))  # find y coordinate of the cell that contains the particle
    # link the previous occupant of the cell to the new. On the first iteration, the previous is empty
    lscl[particle] = cell_array_head[X_cell_ind, Y_cell_ind]
    # the new one goes to the top of the list, to the header:
    cell_array_head[X_cell_ind, Y_cell_ind] = particle
    # print(X_cell_ind, Y_cell_ind)
Lcx=89
Lcy=44

for X_cell_ind in range(Lcx):
    for Y_cell_ind in range(Lcy):
        # scan neighboring cells for each cell (including itself: X_cell_ind, Y_cell_ind)
        for X_cell_nn_ind in (X_cell_ind - 1,X_cell_ind, X_cell_ind + 1):
            for Y_cell_nn_ind in (Y_cell_ind - 1,Y_cell_ind, Y_cell_ind + 1):
                if ((Y_cell_ind == 21)&(X_cell_ind == 88)&(X_cell_nn_ind == 89)&(Y_cell_nn_ind == 21)):
                    print('sdf')
                # skip if Y_cell_nn_ind is out of bounds, since these are outside of the box.
                if ((Y_cell_nn_ind >= 0) & (Y_cell_nn_ind < Lcy)):  # -1 and Lcy are Out of bounds
                    # periodic boundary conditions in x
                    y_shift = 0
                    x_shift = 0
                    X_cell_nn_ind_use = 0+ X_cell_nn_ind
                    if X_cell_nn_ind_use == -1:  # out of bounds
                        X_cell_nn_ind_use += Lcx
                        x_shift = BoxX  ######## changed sign
                    if X_cell_nn_ind_use == Lcx:  # out of bounds
                        X_cell_nn_ind_use -= Lcx
                        x_shift = -BoxX  ######## changed sign
                    # Scan first particle in cell X_cell_ind, Y_cell_ind
                    l = (cell_array_head[X_cell_ind, Y_cell_ind])
                    # scan all the particles in cell X_cell_ind, Y_cell_ind
                    while ~np.isnan(l):
                        # scan particle m in neighboring cell (including in cell X_cell_ind, Y_cell_ind)
                        m = (cell_array_head[X_cell_nn_ind_use, Y_cell_nn_ind])
                        while ~np.isnan(m):
                            # m=int(m)
                            if (l < m):  # avoid double counting.
                                # Get coordinates x and y.
                                x_l = coordsForRunning[jo, int(l * 3 + 1)]
                                x_m = coordsForRunning[jo, int(m * 3 + 1)]
                                y_l = coordsForRunning[jo, int(l * 3 + 2)]
                                y_m = coordsForRunning[jo, int(m * 3 + 2)]
                                # Shift due to periodic conditions:
                                x_lm = x_l - (x_m - x_shift)
                                y_lm = y_l - (y_m - y_shift)
                                r_lm_sqrd = x_lm ** 2 + y_lm ** 2
                                if r_lm_sqrd < IntRange ** 2:
                                    # compute forces on pair (l,m):
                                    Force_X, Force_Y = f_pp((x_l, y_l), (x_m, y_m), epsilon, IntRange, radius,
                                                            ppType=ppType, x_shift=x_shift,
                                                            y_shift=y_shift)  # will also shift due to periodic conditions
                                    print(Force_X,l,m)
                                    coordsForRunning[jn, int(l * 3 + 1)] += Force_X / drag_N_sPum * dt
                                    coordsForRunning[jn, int(l * 3 + 2)] += Force_Y / drag_N_sPum * dt
                                    coordsForRunning[jn, int(m * 3 + 1)] -= Force_X / drag_N_sPum * dt
                                    coordsForRunning[jn, int(m * 3 + 2)] -= Force_Y / drag_N_sPum * dt
                                    # apply periodic boundary conditions on updated particles- X axis ONLY.
                                    for int_prtcls in [l, m]:
                                        # if coordsForRunning[jn,int(int_prtcls*3+1)]<0:
                                        #     coordsForRunning[jn,int(int_prtcls*3+1)]+=BoxX
                                        # if coordsForRunning[jn,int(int_prtcls*3+1)]>BoxX:
                                        #     coordsForRunning[jn,int(int_prtcls*3+1)]-=BoxX
                                        # Useful warning flag on out of bounds cases. However, will not work with python2 and Numba. Wors in python3.
                                        if ((coordsForRunning[jn, int(int_prtcls * 3 + 2)] > BoxY) | (
                                                coordsForRunning[jn, int(int_prtcls * 3 + 2)] < 0)):
                                            print(i, 'y: ', coordsForRunning[jo, int(int_prtcls * 3 + 2)], 'x: ',
                                                  coordsForRunning[jo, int(int_prtcls * 3 + 1)], 'r: ',
                                                  sqrt(r_lm_sqrd), 'Y translation', Force_Y / drag_N_sPum * dt,
                                                  'X translation', Force_X / drag_N_sPum * dt, 'Particle', int_prtcls,
                                                  'Lcx', X_cell_nn_ind, X_cell_ind, 'X_shift', x_shift)
                                            # WarningStr=('timestep='+str(i)+' translationY_um= '+str(TranslationY_um)+
                                            #             'um. r_lm= '+ str(r_lm)+'g='+ str(g)+
                                            #             'y_12= '+ str(y_12)+'x_12= '+ str(x_12)+
                                            #             'epsilon'+ str(epsilon)+'drag_N_sPum'+str(drag_N_sPum)+
                                            #             'dt', str(dt))
                                            # print(WarningStr)
                            m = lscl[int(m)]  # get the next particle in the neighboring cell.
                        l = lscl[int(l)]  # get the next particle in X_cell_ind, Y_cell_ind cell.
