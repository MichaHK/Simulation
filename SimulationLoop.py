from numba import jit, prange
import numpy as np
from math import sqrt, ceil, floor
from forces import f_pp, wall_force_linear, exp_wall_force,wall_force_biharm
import matplotlib.pyplot as plt


# @jit(nopython=True)  # USE THIS, parallel=True)
# def runSimulationPPInt(coordsForProfile, coordsForRunning, velocity, NumberOfParticles,
#                        dt, numberOfSteps, NumberOfSamplesForDensityProfile,
#                        drag_N_sPum, ForceFactor, ChannelLength_um,
#                        radius, BoxX, BoxY, IntRange, D_T, D_R, y_wl, y_wr, f_max_N, epsilon, ppType='WCA', WallType = 'linear'):
#     # initialize the variables for the linked cells.
#     sigma = 2 * radius
#     Lcx = int(floor(BoxX / IntRange))  # Number of x cells, # BoxX is Region[0] in the original script;
#     Lcy = int(floor(BoxY / IntRange))  # Number of y cells, BoxY is Region[1] in the original script;
#     IntRange_cx = BoxX / Lcx  # edge in um of x cell. rc[0] in original script
#     IntRange_cy = BoxY / Lcy  # edge in um of y cell. rc[1] in original script
#     lscl = np.zeros(NumberOfParticles, np.float_)
#     cell_array_head = np.zeros((Lcx, Lcy),
#                                np.float_)  # array with all cells. Will point to the first particle in each cell.
#     # noise: initial noise for first 10000 steps
#     x_noises_um = sqrt(2 * D_T * dt) * np.random.randn(10000, NumberOfParticles) + 0
#     y_noises_um = sqrt(2 * D_T * dt) * np.random.randn(10000, NumberOfParticles) + 0
#     p_noises = sqrt(2 * D_R * dt) * np.random.randn(10000, NumberOfParticles) + 0
#     # For export array:
#     LastFrameToUse = numberOfSteps - (numberOfSteps % NumberOfSamplesForDensityProfile)
#     residual = numberOfSteps % NumberOfSamplesForDensityProfile
#     gap = ceil(LastFrameToUse / NumberOfSamplesForDensityProfile)
#     k = 0  # indicator for export array.
#     coordsForProfile[0, :] = coordsForRunning[0, :]
#     # Run over timesteps
#     for i in range(1, numberOfSteps):
#         # indexes for the two rows containing old and new coordinates (coordsForRunning)
#         jn = i % 2
#         jo = (i - 1) % 2
#         # progress bar
#         TenPercent = int(numberOfSteps / 10)
#         if i % TenPercent == 0:
#             print(i / numberOfSteps * 100, '%')
#         # resetting the head and list for the linked cell lists (will be used for inter-particle interactions):
#         cell_array_head[:, :] = np.nan
#         lscl[:] = np.nan
#         # generate noise every 10000 steps
#         Noise_ind = i % 10000
#         if Noise_ind == 0:
#             x_noises_um = sqrt(2 * D_T * dt) * np.random.randn(10000, NumberOfParticles) + 0
#             y_noises_um = sqrt(2 * D_T * dt) * np.random.randn(10000, NumberOfParticles) + 0
#             p_noises = sqrt(2 * D_R * dt) * np.random.randn(10000, NumberOfParticles) + 0
#         # running over all particles for diffusion, velocity and also to form the linked cell list.
#         for particle in range(NumberOfParticles):  # should be prange
#
#             # Adding contribution due to Diffusion and Velocity:
#             coordsForRunning[jn, particle * 3] = coordsForRunning[jo, particle * 3] + p_noises[
#                 Noise_ind, particle]  # p coordinate of particle (angle)
#             coordsForRunning[jn, particle * 3 + 1] = coordsForRunning[jo, particle * 3 + 1] + x_noises_um[
#                 Noise_ind, particle] + velocity * np.cos(
#                 coordsForRunning[jo, particle * 3]) * dt  # x coordinate of particle (angle)
#             coordsForRunning[jn, particle * 3 + 2] = coordsForRunning[jo, particle * 3 + 2] + y_noises_um[
#                 Noise_ind, particle] + velocity * np.sin(
#                 coordsForRunning[jo, particle * 3]) * dt  # x coordinate of particle (angle)
#
#             # Adding contribution due to wall forces
#             if coordsForRunning[jo, particle * 3 + 2] < y_wl:
#                 coordsForRunning[jn, particle * 3 + 2] += (f_max_N - (f_max_N / y_wl) * coordsForRunning[
#                     jo, particle * 3 + 2]) / drag_N_sPum * ForceFactor * dt
#             if coordsForRunning[jo, particle * 3 + 2] > y_wr:
#                 coordsForRunning[jn, particle * 3 + 2] += -(f_max_N / (y_wr - ChannelLength_um) * (
#                         y_wr - coordsForRunning[jo, particle * 3 + 2])) / drag_N_sPum * ForceFactor * dt
#
#             # Build linked cell list for timestep i, to be used for inter-particle forces
#             # http://cacs.usc.edu/education/cs596/01-1LinkedListCell.pdf
#             X_cell_ind = int(floor(coordsForRunning[
#                                        jo, particle * 3 + 1] / IntRange_cx))  # find x coordinate of the cell that contains the particle
#             Y_cell_ind = int(floor(coordsForRunning[
#                                        jo, particle * 3 + 2] / IntRange_cy))  # find y coordinate of the cell that contains the particle
#             # link the previous occupant of the cell to the new. On the first iteration, the previous is empty
#             lscl[particle] = cell_array_head[X_cell_ind, Y_cell_ind]
#             # the new one goes to the top of the list, to the header:
#             cell_array_head[X_cell_ind, Y_cell_ind] = particle
#
#             # Apply the periodic boundary conditions on the new positions.
#             # I also apply again on every particle I move in the pp interaction loop.
#             # if coordsForRunning[jn, particle*3+1] > BoxX:
#             #     coordsForRunning[jn, particle*3+1] -= BoxX
#             # if coordsForRunning[jn, particle*3+1] < 0:
#             #     coordsForRunning[jn, particle*3+1] += BoxX
#
#         # compute inter-particle interactions for timestep i, all particles.
#         # going over each cell.
#         for X_cell_ind in range(Lcx):
#             for Y_cell_ind in range(Lcy):
#                 # scan neighboring cells for each cell (including itself: X_cell_ind, Y_cell_ind)
#                 for X_cell_nn_ind in (X_cell_ind - 1, X_cell_ind, X_cell_ind + 1):
#                     for Y_cell_nn_ind in (Y_cell_ind - 1, Y_cell_ind, Y_cell_ind + 1):
#                         # skip if Y_cell_nn_ind is out of bounds, since these are outside of the box.
#                         if ((Y_cell_nn_ind >= 0) & (Y_cell_nn_ind < Lcy)):  # -1 and Lcy are Out of bounds
#                             # periodic boundary conditions in x
#                             y_shift = 0
#                             x_shift = 0
#                             X_cell_nn_ind_use = X_cell_nn_ind  # you should not update a looping variable, when there are inner loops.
#                             if X_cell_nn_ind_use == -1:  # out of bounds
#                                 X_cell_nn_ind_use = X_cell_nn_ind + Lcx
#                                 x_shift = BoxX
#                             if X_cell_nn_ind_use == Lcx:  # out of bounds
#                                 X_cell_nn_ind_use = X_cell_nn_ind - Lcx
#                                 x_shift = -BoxX
#                                 # Scan first particle in cell X_cell_ind, Y_cell_ind
#                             l = cell_array_head[X_cell_ind, Y_cell_ind]
#                             # scan all the particles in cell X_cell_ind, Y_cell_ind
#                             while ~np.isnan(l):
#                                 # scan particle m in neighboring cell (including in cell X_cell_ind, Y_cell_ind)
#                                 m = cell_array_head[X_cell_nn_ind_use, Y_cell_nn_ind]
#                                 while ~np.isnan(m):
#                                     # m=int(m)
#                                     if (l < m):  # avoid double counting.
#                                         # Get coordinates x and y.
#                                         x_l = coordsForRunning[jo, int(l * 3 + 1)] + 0
#                                         x_m = coordsForRunning[jo, int(m * 3 + 1)] + 0
#                                         y_l = coordsForRunning[jo, int(l * 3 + 2)] + 0
#                                         y_m = coordsForRunning[jo, int(m * 3 + 2)] + 0
#                                         # Shift due to periodic conditions:
#                                         x_lm = x_l - (x_m - x_shift)
#                                         y_lm = y_l - (y_m - y_shift)
#                                         r_lm_sqrd = x_lm ** 2 + y_lm ** 2
#                                         if r_lm_sqrd < IntRange ** 2:
#                                             # compute forces on pair (l,m):
#                                             Force_X, Force_Y = f_pp((x_l, y_l), (x_m, y_m), epsilon, IntRange, radius,
#                                                                     ppType=ppType, x_shift=x_shift, y_shift=y_shift)
#                                             # r_lm=sqrt(r_lm_sqrd)
#                                             # g=-4*epsilon*(-12/sigma*(sigma/r_lm)**13+6/sigma*(sigma/r_lm)**7)
#                                             # for Particle1,Particle2,x_12,y_12,coordinate in [(l,m,x_lm,y_lm,1),(l,m,x_lm,y_lm,2)]:
#                                             #     TranslationX_um=g*(x_12)/r_lm/drag_N_sPum*dt
#                                             #     TranslationY_um=g*(y_12)/r_lm/drag_N_sPum*dt
#                                             coordsForRunning[jn, int(l * 3 + 1)] += Force_X / drag_N_sPum * dt
#                                             coordsForRunning[jn, int(l * 3 + 2)] += Force_Y / drag_N_sPum * dt
#                                             coordsForRunning[jn, int(m * 3 + 1)] -= Force_X / drag_N_sPum * dt
#                                             coordsForRunning[jn, int(m * 3 + 2)] -= Force_Y / drag_N_sPum * dt
#                                             # apply periodic boundary conditions on updated particles- X axis ONLY.
#                                             # for int_prtcls in [l, m]:
#                                             # if coordsForRunning[jn,int(int_prtcls*3+1)]<0:
#                                             #     coordsForRunning[jn,int(int_prtcls*3+1)]+=BoxX
#                                             # if coordsForRunning[jn,int(int_prtcls*3+1)]>BoxX:
#                                             #     coordsForRunning[jn,int(int_prtcls*3+1)]-=BoxX
#                                             # Useful warning flag on out of bounds cases. However, will not work with python2 and Numba. Works in python3.
#                                             for int_prtcls in (l, m):
#                                                 if ((coordsForRunning[jn, int(int_prtcls * 3 + 2)] > BoxY) | (
#                                                         coordsForRunning[jn, int(int_prtcls * 3 + 2)] < 0)):
#                                                     print(i, 'y: ', coordsForRunning[jo, int(int_prtcls * 3 + 2)],
#                                                           'x: ', coordsForRunning[jo, int(int_prtcls * 3 + 1)], 'r: ',
#                                                           sqrt(r_lm_sqrd), 'Y translation', Force_Y / drag_N_sPum * dt,
#                                                           'X translation', Force_X / drag_N_sPum * dt, 'Particle',
#                                                           int_prtcls,
#                                                           'cell, nn', X_cell_ind, X_cell_nn_ind, 'In use nn cell',
#                                                           X_cell_nn_ind_use, 'X_shift', x_shift, l, m)
#                                     m = lscl[int(m)]  # get the next particle in the neighboring cell.
#                                 l = lscl[int(l)]  # get the next particle in X_cell_ind, Y_cell_ind cell.
#         for particle in range(NumberOfParticles):  # should be prange
#             if coordsForRunning[jn, particle * 3 + 1] > BoxX:
#                 coordsForRunning[jn, particle * 3 + 1] -= BoxX
#             if coordsForRunning[jn, particle * 3 + 1] < 0:
#                 coordsForRunning[jn, particle * 3 + 1] += BoxX
#
#         # Update profile data for export
#
#         if i >= residual:
#             UseForProfile = ((i + residual - 1) % gap == 0)
#             if UseForProfile:
#                 coordsForProfile[k, 0::3] = coordsForRunning[jo, 0::3]
#                 coordsForProfile[k, 1::3] = coordsForRunning[jo, 1::3]
#                 coordsForProfile[k, 2::3] = coordsForRunning[jo, 2::3]
#                 k += 1
#     print('Done')
#     return coordsForProfile
#
#
# @jit(nopython=True)  # USE THIS, parallel=True)
# def runSimulationNoPPInt(coordsForProfile, coordsForRunning, velocity, NumberOfParticles,
#                        dt, numberOfSteps, NumberOfSamplesForDensityProfile,
#                        drag_N_sPum, ForceFactor, ChannelLength_um,
#                        radius, BoxX, BoxY, IntRange, D_T, D_R, y_wl, y_wr, f_max_N, epsilon, ppType='WCA', WallType = 'linear'):
#     sigma = 2 * radius
#     # noise: initial noise for first 10000 steps
#     x_noises_um = sqrt(2 * D_T * dt) * np.random.randn(10000, NumberOfParticles) + 0
#     y_noises_um = sqrt(2 * D_T * dt) * np.random.randn(10000, NumberOfParticles) + 0
#     p_noises = sqrt(2 * D_R * dt) * np.random.randn(10000, NumberOfParticles) + 0
#     # For export array:
#     LastFrameToUse = numberOfSteps - (numberOfSteps % NumberOfSamplesForDensityProfile)
#     residual = numberOfSteps % NumberOfSamplesForDensityProfile
#     gap = ceil(LastFrameToUse / NumberOfSamplesForDensityProfile)
#     k = 0  # indicator for export array.
#     coordsForProfile[0, :] = coordsForRunning[0, :]
#     # Run over timesteps
#     for i in range(1, numberOfSteps):
#         # indexes for the two rows containing old and new coordinates (coordsForRunning)
#         jn = i % 2
#         jo = (i - 1) % 2
#         # progress bar
#         TenPercent = int(numberOfSteps / 10)
#         if i % TenPercent == 0:
#             print(i / numberOfSteps * 100, '%')
#         # generate noise every 10000 steps
#         Noise_ind = i % 10000
#         if Noise_ind == 0:
#             x_noises_um = sqrt(2 * D_T * dt) * np.random.randn(10000, NumberOfParticles) + 0
#             y_noises_um = sqrt(2 * D_T * dt) * np.random.randn(10000, NumberOfParticles) + 0
#             p_noises = sqrt(2 * D_R * dt) * np.random.randn(10000, NumberOfParticles) + 0
#         # running over all particles for diffusion, velocity and also to form the linked cell list.
#         for particle in range(NumberOfParticles):  # should be prange
#
#             # Adding contribution due to Diffusion and Velocity:
#             coordsForRunning[jn, particle * 3] = coordsForRunning[jo, particle * 3] + p_noises[
#                 Noise_ind, particle]  # p coordinate of particle (angle)
#             coordsForRunning[jn, particle * 3 + 1] = coordsForRunning[jo, particle * 3 + 1] + x_noises_um[
#                 Noise_ind, particle] + velocity * np.cos(
#                 coordsForRunning[jo, particle * 3]) * dt  # x coordinate of particle (angle)
#             coordsForRunning[jn, particle * 3 + 2] = coordsForRunning[jo, particle * 3 + 2] + y_noises_um[
#                 Noise_ind, particle] + velocity * np.sin(
#                 coordsForRunning[jo, particle * 3]) * dt  # x coordinate of particle (angle)
#
#             # Adding contribution due to wall forces
#             if coordsForRunning[jo, particle * 3 + 2] < y_wl:
#                 coordsForRunning[jn, particle * 3 + 2] += (f_max_N - (f_max_N / y_wl) * coordsForRunning[
#                     jo, particle * 3 + 2]) / drag_N_sPum * ForceFactor * dt
#             if coordsForRunning[jo, particle * 3 + 2] > y_wr:
#                 coordsForRunning[jn, particle * 3 + 2] += -(f_max_N / (y_wr - ChannelLength_um) * (
#                         y_wr - coordsForRunning[jo, particle * 3 + 2])) / drag_N_sPum * ForceFactor * dt
#
#
#         for particle in range(NumberOfParticles):  # should be prange
#             if coordsForRunning[jn, particle * 3 + 1] > BoxX:
#                 coordsForRunning[jn, particle * 3 + 1] -= BoxX
#             if coordsForRunning[jn, particle * 3 + 1] < 0:
#                 coordsForRunning[jn, particle * 3 + 1] += BoxX
#
#         # Update profile data for export
#
#         if i >= residual:
#             UseForProfile = ((i + residual - 1) % gap == 0)
#             if UseForProfile:
#                 coordsForProfile[k, 0::3] = coordsForRunning[jo, 0::3]
#                 coordsForProfile[k, 1::3] = coordsForRunning[jo, 1::3]
#                 coordsForProfile[k, 2::3] = coordsForRunning[jo, 2::3]
#                 k += 1
#     print('Done')
#     return coordsForProfile


@jit(nopython=True)  # USE THIS, parallel=True)
def runSimulationTorquePPInt(coordsForProfile, coordsForRunning, velocity, NumberOfParticles,
                               dt, numberOfSteps, NumberOfSamplesForDensityProfile,
                               drag_N_sPum, ForceFactor, ChannelLength_um,
                               radius, BoxX, BoxY, IntRange, D_T, D_R, y_wl, y_wr, f_max_N,
                               epsilon, ppType,
                               TorqueFactor, arm, rot_drag_N_sPum3, WallType = 'linear'):
    sigma = 2 * radius
    # initialize the variables for the linked cells.
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
    # For export array:
    LastFrameToUse = numberOfSteps - (numberOfSteps % NumberOfSamplesForDensityProfile)
    residual = numberOfSteps % NumberOfSamplesForDensityProfile
    gap = ceil(LastFrameToUse / NumberOfSamplesForDensityProfile)
    k = 0  # indicator for export array.
    coordsForProfile[0, :] = coordsForRunning[0, :]
    # Run over timesteps
    for i in range(1, numberOfSteps):
        # indexes for the two rows containing old and new coordinates (coordsForRunning)
        jn = i % 2
        jo = (i - 1) % 2
        # progress bar
        TenPercent = int(numberOfSteps / 10)
        if i % TenPercent == 0:
            print(i / numberOfSteps * 100, '%')
        # resetting the head and list for the linked cell lists (will be used for inter-particle interactions):
        cell_array_head[:, :] = np.nan
        lscl[:] = np.nan
        # generate noise every 10000 steps
        Noise_ind = i % 10000
        if Noise_ind == 0:
            x_noises_um = sqrt(2 * D_T * dt) * np.random.randn(10000, NumberOfParticles) + 0
            y_noises_um = sqrt(2 * D_T * dt) * np.random.randn(10000, NumberOfParticles) + 0
            p_noises = sqrt(2 * D_R * dt) * np.random.randn(10000, NumberOfParticles) + 0
        # running over all particles for diffusion, velocity and also to form the linked cell list.
        for particle in range(NumberOfParticles):  # should be prange
            SinTheta = np.sin(coordsForRunning[jo, particle * 3])
            CosTheta = np.cos(coordsForRunning[jo, particle * 3])
            # Adding contribution due to Diffusion and Velocity:
            coordsForRunning[jn, particle * 3] = coordsForRunning[jo, particle * 3] + p_noises[
                Noise_ind, particle]  # p coordinate of particle (angle)
            coordsForRunning[jn, particle * 3 + 1] = coordsForRunning[jo, particle * 3 + 1] + x_noises_um[
                Noise_ind, particle] + velocity * CosTheta * dt  # x coordinate of particle (angle)
            coordsForRunning[jn, particle * 3 + 2] = coordsForRunning[jo, particle * 3 + 2] + y_noises_um[
                Noise_ind, particle] + velocity * SinTheta * dt  # x coordinate of particle (angle)

            # Adding contribution due to wall forces ### 5/3/19
            if WallType == 'linear':
                f = wall_force_linear(coordsForRunning[jo, particle * 3 + 2], f_max_N, y_wl, y_wr, ChannelLength_um)
            if WallType == 'experimental':
                f = exp_wall_force(coordsForRunning[jo, particle * 3 + 2], ChannelLength_um)
            coordsForRunning[jn, particle * 3 + 2] += f / drag_N_sPum * ForceFactor * dt
            # Adding wall torque
            if TorqueFactor != 0:
                if WallType == 'linear':
                    f_at_hem = wall_force_linear(coordsForRunning[jo, particle * 3 + 2] + arm * SinTheta, f_max_N, y_wl, y_wr, ChannelLength_um)
                if WallType == 'experimental':
                    f_at_hem = exp_wall_force(coordsForRunning[jo, particle * 3 + 2] + arm * SinTheta, ChannelLength_um)
                dTheta = f_at_hem * TorqueFactor * arm * CosTheta / rot_drag_N_sPum3 * ForceFactor * dt
                coordsForRunning[jn, particle * 3] += dTheta
                if np.abs(dTheta) > 0.2:
                    print('dTheta is', dTheta, 'at timestep' , i, 'and y location', coordsForRunning[jn, particle * 3 + 2])

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

        # compute inter-particle interactions for timestep i, all particles.
        # going over each cell.
        for X_cell_ind in range(Lcx):
            for Y_cell_ind in range(Lcy):
                # scan neighboring cells for each cell (including itself: X_cell_ind, Y_cell_ind)
                for X_cell_nn_ind in (X_cell_ind - 1, X_cell_ind, X_cell_ind + 1):
                    for Y_cell_nn_ind in (Y_cell_ind - 1, Y_cell_ind, Y_cell_ind + 1):
                        # skip if Y_cell_nn_ind is out of bounds, since these are outside of the box.
                        if ((Y_cell_nn_ind >= 0) & (Y_cell_nn_ind < Lcy)):  # -1 and Lcy are Out of bounds
                            # periodic boundary conditions in x
                            y_shift = 0
                            x_shift = 0
                            X_cell_nn_ind_use = X_cell_nn_ind  # you should not update a looping variable, when there are inner loops.
                            if X_cell_nn_ind_use == -1:  # out of bounds
                                X_cell_nn_ind_use = X_cell_nn_ind + Lcx
                                x_shift = BoxX
                            if X_cell_nn_ind_use == Lcx:  # out of bounds
                                X_cell_nn_ind_use = X_cell_nn_ind - Lcx
                                x_shift = -BoxX
                                # Scan first particle in cell X_cell_ind, Y_cell_ind
                            l = cell_array_head[X_cell_ind, Y_cell_ind]
                            # scan all the particles in cell X_cell_ind, Y_cell_ind
                            while ~np.isnan(l):
                                # scan particle m in neighboring cell (including in cell X_cell_ind, Y_cell_ind)
                                m = cell_array_head[X_cell_nn_ind_use, Y_cell_nn_ind]
                                while ~np.isnan(m):
                                    # m=int(m)
                                    if (l < m):  # avoid double counting.
                                        # Get coordinates x and y.
                                        x_l = coordsForRunning[jo, int(l * 3 + 1)] + 0
                                        x_m = coordsForRunning[jo, int(m * 3 + 1)] + 0
                                        y_l = coordsForRunning[jo, int(l * 3 + 2)] + 0
                                        y_m = coordsForRunning[jo, int(m * 3 + 2)] + 0
                                        # Shift due to periodic conditions:
                                        x_lm = x_l - (x_m - x_shift)
                                        y_lm = y_l - (y_m - y_shift)
                                        r_lm_sqrd = x_lm ** 2 + y_lm ** 2
                                        if r_lm_sqrd < IntRange ** 2:
                                            # compute forces on pair (l,m):
                                            Force_X, Force_Y = f_pp((x_l, y_l), (x_m, y_m), epsilon, IntRange, radius,
                                                                    ppType=ppType, x_shift=x_shift, y_shift=y_shift)
                                            # r_lm=sqrt(r_lm_sqrd)
                                            # g=-4*epsilon*(-12/sigma*(sigma/r_lm)**13+6/sigma*(sigma/r_lm)**7)
                                            # for Particle1,Particle2,x_12,y_12,coordinate in [(l,m,x_lm,y_lm,1),(l,m,x_lm,y_lm,2)]:
                                            #     TranslationX_um=g*(x_12)/r_lm/drag_N_sPum*dt
                                            #     TranslationY_um=g*(y_12)/r_lm/drag_N_sPum*dt
                                            coordsForRunning[jn, int(l * 3 + 1)] += Force_X / drag_N_sPum * dt
                                            coordsForRunning[jn, int(l * 3 + 2)] += Force_Y / drag_N_sPum * dt
                                            coordsForRunning[jn, int(m * 3 + 1)] -= Force_X / drag_N_sPum * dt
                                            coordsForRunning[jn, int(m * 3 + 2)] -= Force_Y / drag_N_sPum * dt
                                            # apply periodic boundary conditions on updated particles- X axis ONLY.
                                            # for int_prtcls in [l, m]:
                                            # if coordsForRunning[jn,int(int_prtcls*3+1)]<0:
                                            #     coordsForRunning[jn,int(int_prtcls*3+1)]+=BoxX
                                            # if coordsForRunning[jn,int(int_prtcls*3+1)]>BoxX:
                                            #     coordsForRunning[jn,int(int_prtcls*3+1)]-=BoxX
                                            # Useful warning flag on out of bounds cases. However, will not work with python2 and Numba. Works in python3.
                                            for int_prtcls in (l, m):
                                                if ((coordsForRunning[jn, int(int_prtcls * 3 + 2)] > BoxY) | (
                                                        coordsForRunning[jn, int(int_prtcls * 3 + 2)] < 0)):
                                                    print(i, 'y: ', coordsForRunning[jo, int(int_prtcls * 3 + 2)],
                                                          'x: ', coordsForRunning[jo, int(int_prtcls * 3 + 1)], 'r: ',
                                                          sqrt(r_lm_sqrd), 'Y translation', Force_Y / drag_N_sPum * dt,
                                                          'X translation', Force_X / drag_N_sPum * dt, 'Particle',
                                                          int_prtcls,
                                                          'cell, nn', X_cell_ind, X_cell_nn_ind, 'In use nn cell',
                                                          X_cell_nn_ind_use, 'X_shift', x_shift, l, m)
                                    m = lscl[int(m)]  # get the next particle in the neighboring cell.
                                l = lscl[int(l)]  # get the next particle in X_cell_ind, Y_cell_ind cell.

        for particle in range(NumberOfParticles):  # should be prange
            if coordsForRunning[jn, particle * 3 + 1] > BoxX:
                coordsForRunning[jn, particle * 3 + 1] -= BoxX
            if coordsForRunning[jn, particle * 3 + 1] < 0:
                coordsForRunning[jn, particle * 3 + 1] += BoxX

        # Update profile data for export

        if i >= residual:
            UseForProfile = ((i + residual - 1) % gap == 0)
            if UseForProfile:
                coordsForProfile[k, 0::3] = coordsForRunning[jo, 0::3]
                coordsForProfile[k, 1::3] = coordsForRunning[jo, 1::3]
                coordsForProfile[k, 2::3] = coordsForRunning[jo, 2::3]
                k += 1
    print('Done')
    return coordsForProfile


@jit(nopython=True, parallel=True)
def runSimulationTorqueNoPPInt(coordsForProfile, coordsForRunning, velocity, NumberOfParticles,
                               dt, numberOfSteps, NumberOfSamplesForDensityProfile,
                               drag_N_sPum, ForceFactor, ChannelLength_um,
                               radius, BoxX, BoxY, IntRange, D_T, D_R, y_wl, y_wr, f_max_N,
                               epsilon, ppType,
                               TorqueFactor, arm, rot_drag_N_sPum3,
                               ReportOrderParameter, Range_um_for_OrderParameterCalc, LocationsToReportOrderParameter,
                               WallType = 'linear'):
    ##
    sigma = 2 * radius
    ## noise: initial noise for first 10000 steps
    x_noises_um = sqrt(2 * D_T * dt) * np.random.randn(10000, NumberOfParticles) + 0
    y_noises_um = sqrt(2 * D_T * dt) * np.random.randn(10000, NumberOfParticles) + 0
    p_noises = sqrt(2 * D_R * dt) * np.random.randn(10000, NumberOfParticles) + 0
    ## For export array:
    LastFrameToUse = numberOfSteps - (numberOfSteps % NumberOfSamplesForDensityProfile)
    residual = numberOfSteps % NumberOfSamplesForDensityProfile
    gap = ceil(LastFrameToUse / NumberOfSamplesForDensityProfile)
    k = 0  # indicator for export array.
    coordsForProfile[0, :] = coordsForRunning[0, :] # p, x, y
    if ReportOrderParameter:
        S = np.zeros(len(LocationsToReportOrderParameter))
        OrderParameterSampleCounters = np.zeros(len(LocationsToReportOrderParameter))
        OrderParameter = np.zeros(len(LocationsToReportOrderParameter))
    ## Run over timesteps
    for i in range(1, numberOfSteps):
        # indexes for the two rows containing old and new coordinates (coordsForRunning)
        jn = i % 2
        jo = (i - 1) % 2
        # progress bar
        TenPercent = int(numberOfSteps / 10)
        if i % TenPercent == 0:
            print(i / numberOfSteps * 100, '%')
        # generate noise every 10000 steps
        Noise_ind = i % 10000
        if Noise_ind == 0:
            x_noises_um = sqrt(2 * D_T * dt) * np.random.randn(10000, NumberOfParticles) + 0
            y_noises_um = sqrt(2 * D_T * dt) * np.random.randn(10000, NumberOfParticles) + 0
            p_noises = sqrt(2 * D_R * dt) * np.random.randn(10000, NumberOfParticles) + 0
        # running over all particles for diffusion, velocity and also to form the linked cell list.
        for particle in range(NumberOfParticles):  # should be prange
            SinTheta = np.sin(coordsForRunning[jo, particle * 3])
            CosTheta = np.cos(coordsForRunning[jo, particle * 3])
            # Adding contribution due to Diffusion and Velocity:
            coordsForRunning[jn, particle * 3] = coordsForRunning[jo, particle * 3] + p_noises[
                Noise_ind, particle]  # p coordinate of particle (angle)
            coordsForRunning[jn, particle * 3 + 1] = coordsForRunning[jo, particle * 3 + 1] + x_noises_um[
                Noise_ind, particle] + velocity * CosTheta * dt  # x coordinate of particle
            coordsForRunning[jn, particle * 3 + 2] = coordsForRunning[jo, particle * 3 + 2] + y_noises_um[
                Noise_ind, particle] + velocity * SinTheta * dt  # x coordinate of particle

            # Adding contribution due to wall forces ### 5/3/19
            if WallType == 'linear':
                f = wall_force_linear(coordsForRunning[jo, particle * 3 + 2], f_max_N, y_wl, y_wr, ChannelLength_um)
            if WallType == 'experimental':
                f = exp_wall_force(coordsForRunning[jo, particle * 3 + 2], ChannelLength_um)
            if WallType == 'BiHarm':
                f = wall_force_biharm(coordsForRunning[jo, particle * 3 + 2], f_max_N, y_wl, y_wr, ChannelLength_um)
            coordsForRunning[jn, particle * 3 + 2] += f / drag_N_sPum * ForceFactor * dt
            # Adding wall torque
            if WallType == 'linear':
                f_at_hem = wall_force_linear(coordsForRunning[jo, particle * 3 + 2] + arm * SinTheta, f_max_N, y_wl, y_wr, ChannelLength_um)
            if WallType == 'experimental':
                f_at_hem = exp_wall_force(coordsForRunning[jo, particle * 3 + 2] + arm * SinTheta, ChannelLength_um)
            dTheta = f_at_hem * TorqueFactor * arm * CosTheta / rot_drag_N_sPum3 * ForceFactor * dt
            # dTheta = 0
            coordsForRunning[jn, particle * 3] += dTheta
            if np.abs(dTheta) > 0.2:
                print('dTheta is', dTheta, 'at timestep' , i, 'and y location', coordsForRunning[jn, particle * 3 + 2])

            if ReportOrderParameter:
                y = coordsForRunning[jo, particle * 3 + 2]
                Theta = coordsForRunning[jn, particle * 3]
                for ind in range(len(LocationsToReportOrderParameter)):
                    if ((LocationsToReportOrderParameter[ind] - y)**2 < Range_um_for_OrderParameterCalc**2):
                        S[ind] = S[ind] + (2*(np.cos(Theta))**2 - 1)
                        OrderParameterSampleCounters[ind] = OrderParameterSampleCounters[ind] + 1

        for particle in range(NumberOfParticles):  # should be prange
            if coordsForRunning[jn, particle * 3 + 1] > BoxX:
                coordsForRunning[jn, particle * 3 + 1] -= BoxX
            if coordsForRunning[jn, particle * 3 + 1] < 0:
                coordsForRunning[jn, particle * 3 + 1] += BoxX

        # Update profile data for export

        if i >= residual:
            UseForProfile = ((i + residual - 1) % gap == 0)
            if UseForProfile:
                # coordsForProfile[k, 0::3] = coordsForRunning[jo, 0::3] # angle
                coordsForProfile[k, 0::3] = coordsForRunning[jo, 0::3] # p
                coordsForProfile[k, 1::3] = coordsForRunning[jo, 1::3] # x
                coordsForProfile[k, 2::3] = coordsForRunning[jo, 2::3] # y
                k += 1
    print('Done')
    if ReportOrderParameter:
        for ind in range(len(S)): # pythonian way not working with jit here
            if OrderParameterSampleCounters[ind] != 0:
                OrderParameter[ind] = S[ind]/ OrderParameterSampleCounters[ind]
            else:
                OrderParameter[ind] = np.nan
    else:
        OrderParameter = np.zeros(len(LocationsToReportOrderParameter))
    return coordsForProfile, OrderParameter, OrderParameterSampleCounters
