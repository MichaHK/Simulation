# try to

### load imported
import time
import os
import numpy as np
from forces import *
from SimulationAux import writeProbDistFunc, CalcPDFForYlocations, writeMetaData, getFilenameBase, \
    getAverageForceAsym
from PlaceInitialParticles import *
from SimulationLoop import runSimulationTorqueNoPPInt, runSimulationTorquePPInt
from math import ceil, floor, sqrt
import matplotlib.pyplot as plt

subFolder = '20190212_dblLineAsym2/With008CalibDragMult2'
SaveCoords = True
### Setting pp interactions
epsilon =  10 ** (-15)
radius = 1
ppType = 'WCA'
###  setting wall interactions
WallType = 'experimental' #'BiHarm'#'linear' #  go into forces.py and make sure you are using the right profile.
Voltage_eq = 0.64
ForceExp = 1  # change it here or meta files will be bad.
f_max_N = 5 * 10 ** (-15)
y_wl = 10
y_wr = 50
RhoZeroLocation_um = np.mean([y_wl, y_wr])
ChannelLength_um = 97
BoxX = ChannelLength_um * 2
### setting simulation params
ActiveVoltages = [10.2]
D_T = 0.1  # use um^2/s, translational
D_Rs = [1/10]
Cs = [1]  # ratio between the
# effective dielectric constants (hematite/TPM). Less than 1 would push wall, more than 1 would scatter off the wall.
NumberOfParticlesArray = [1]
dts = [0.05]
numberOfSteps_s = [9000000]

velocities = [2.7]
drag_multiplier = 2
kT = 4.11 * 10 ** (-21)
bins = 400

################################################################################

### init pp interactions
if ppType == 'WCA':
    IntRange = 2 * radius * 2 ** (1 / 6)
if ppType == 'Harmonic':
    IntRange = 2 * radius
### init wall interactions
dist_Eq = np.linspace(0, ChannelLength_um, ChannelLength_um + 1)
BoxY = ChannelLength_um  # else the np.mod in brwonian motion would not work)
if WallType == 'linear':
    Force_Eq = np.array([wall_force_linear(y, f_max_N, y_wl, y_wr, ChannelLength_um) for y in dist_Eq])
elif WallType == 'BiHarm':
    Force_Eq = np.array([wall_force_biharm(y, f_max_N, y_wl, y_wr, ChannelLength_um) for y in dist_Eq])
elif WallType == 'experimental':
    Force_Eq = np.array([exp_wall_force(y,ChannelLength_um) for y in dist_Eq])

### init wall torques
L_H = 0.8                               # hematite length
# C = 1.1                                 # ratio between the effective dielectric constants (hematite/TPM). Less than 1 would push wall, more than 1 would scatter off the wall.
V_H = L_H**3                            # hematite volume
V_T = 0.333333*4*np.pi*radius**3-V_H    # TPM volume
arm = radius - 0.5*L_H
rot_drag_N_sPum3 = kT / (8*np.pi*radius**3) * 10 ** 6

### folder to save
if not os.path.isdir(subFolder):
    os.mkdir(subFolder)  # creating new folder. WRONG WAY: use os.mkdir instead so will work on WSL as well.
CurrentFolder = os.getcwd()
CurrentFolder = CurrentFolder.replace('\\', r'/') + '/' + subFolder
CurrentFolder = CurrentFolder.replace('/notebooks', '')

# run the simulation
drag_N_sPum = drag_multiplier * kT / D_T * 10 ** 6
for C in Cs:
    TorqueFactor=(C-1)/(C*V_H+V_T) * V_H
    if TorqueFactor==0:
        print('Will skip torque')
    for D_R in D_Rs:
        for ActiveVoltage in ActiveVoltages:
            ForceFactor = 1 * (ActiveVoltage / Voltage_eq) ** 2
            for velocity in velocities:
                if velocity < Force_Eq.max() * ForceFactor / drag_N_sPum * (C*2):
                    for ctr1, NumberOfParticles in enumerate(NumberOfParticlesArray):
                        print('run...')
                        # setting dt, number of steps and sampling intervals for simulation
                        dt=dts[ctr1]
                        numberOfSteps=numberOfSteps_s[ctr1]
                        SamplingTimeForDensityProfile = 0.2 #radius / (velocity / D_R) / D_R/ 2
                        NumberOfSamplesForDensityProfile = ceil(numberOfSteps * dt / SamplingTimeForDensityProfile)
                        # initializing the exported sample coordinates.
                        coordsForProfile = np.empty([NumberOfSamplesForDensityProfile,
                                                     3 * NumberOfParticles])  # p1x1y1 (particle 1), p2x2y2 (particle 2),.... [array] p is the angle. Each row is a timestep.
                        # coordsForProfile = np.empty([NumberOfSamplesForDensityProfile,
                        #                              4 * NumberOfParticles])  # x1y1dx1dy1 (particle 1), x2y2dx2dy2 (particle 2),.... [array]. Each row is a timestep.
                        coordsForRunning = np.empty([2,
                                                     3 * NumberOfParticles])  # p1x1y1 (particle 1), p2x2y2 (particle 2),.... [array] p is the angle. Each row is a timestep.
                        # setting initial conditions: positions, velocities.
                        InitialPts = GetInitialPts(NumberOfParticles, BoxX, BoxY)
                        coordsForRunning[0, 1::3] = np.array(InitialPts)[:, 0]  # inital x coordinates.
                        coordsForRunning[0, 2::3] = np.array(InitialPts)[:, 1]  # inital y coordinates.
                        coordsForRunning[0, 0::3] = 2*3.14159*np.random.rand(len(InitialPts)) # sqrt(2 * D_R * dt) * np.random.randn(len(InitialPts)) + 0
                        # check number of bins
                        if ChannelLength_um/bins > velocity*dt:
                            print('Histogram uses ', bins, 'bins for a channel length of',ChannelLength_um,', which is maybe too little')

                        # run the loops
                        t = time.time()
                        # coords = runSimulationNoPPInt(coordsForProfile, coordsForRunning, velocity, NumberOfParticles,
                        #                             dt, numberOfSteps, NumberOfSamplesForDensityProfile,
                        #                             drag_N_sPum, ForceFactor, ChannelLength_um,
                        #                             radius, BoxX, BoxY, IntRange, D_T, D_R, y_wl, y_wr, f_max_N,
                        #                             epsilon, ppType)

                        # coords = runSimulationTorquePPInt(coordsForProfile, coordsForRunning, velocity, NumberOfParticles,
                        #                             dt, numberOfSteps, NumberOfSamplesForDensityProfile,
                        #                             drag_N_sPum, ForceFactor, ChannelLength_um,
                        #                             radius, BoxX, BoxY, IntRange, D_T, D_R, y_wl, y_wr, f_max_N,
                        #                             epsilon, ppType, TorqueFactor, arm, rot_drag_N_sPum3, WallType = WallType)

                        coords = runSimulationTorqueNoPPInt(coordsForProfile, coordsForRunning, velocity, NumberOfParticles,
                                                    dt, numberOfSteps, NumberOfSamplesForDensityProfile,
                                                    drag_N_sPum, ForceFactor, ChannelLength_um,
                                                    radius, BoxX, BoxY, IntRange, D_T, D_R, y_wl, y_wr, f_max_N,
                                                    epsilon, ppType, TorqueFactor, arm, rot_drag_N_sPum3, WallType = WallType)


                        elapsed = time.time() - t
                        print(elapsed)

                        ## exporting
                        # getting histographs
                        x_hist, hist = CalcPDFForYlocations(coords[:, 2::3], bins=bins)
                        # only use the second hald for the histogram:
                        x_hist_lastHalf, hist_lastHalf = CalcPDFForYlocations(coords[int(len(coords)/2):,2::3], bins=bins)
                        indToRhoZero = np.argmin(np.abs(x_hist - RhoZeroLocation_um))
                        ProbAtCenter = hist[indToRhoZero] # REALLY BAD. I need an aux function to do cumsum to the force function and find the max point. That is where the Prob should be sampled.
                        # calculate pressure
                        AverageForceLeft, AverageForceRight, SumAverageForce = getAverageForceAsym(x_hist, hist,
                                                                                                   ForceFactor,
                                                                                                   f_max_N, y_wl, y_wr,
                                                                                                   ChannelLength_um, WallType=WallType)
                        AverageForceLeft_lastHalf, AverageForceRight_lastHalf, SumAverageForce_lastHalf = getAverageForceAsym(x_hist_lastHalf, hist_lastHalf,
                                                                                                                              ForceFactor,
                                                                                                                              f_max_N, y_wl, y_wr,
                                                                                                                              ChannelLength_um, WallType=WallType)
                        PressureLeft_NPum, PressureRight_NPum, SumPressure_NPum = np.array(
                            [AverageForceLeft, AverageForceRight, SumAverageForce]) * NumberOfParticles / BoxX
                        PressureLeft_NPum_lastHalf, PressureRight_NPum_lastHalf, SumPressure_NPum_lastHalf = np.array(
                            [AverageForceLeft_lastHalf, AverageForceRight_lastHalf, SumAverageForce_lastHalf]) * NumberOfParticles / BoxX
                        # print(AverageForceLeft,AverageForceRight,SumAverageForce)
                        print(PressureLeft_NPum, PressureRight_NPum, SumPressure_NPum)
                        # export
                        filenameBase = getFilenameBase(ActiveVoltage, velocity, D_T, D_R, NumberOfParticles,
                                                       drag_multiplier, C)
                        writeMetaData(filenameBase, kT, drag_N_sPum, drag_multiplier, dt, numberOfSteps,
                                      NumberOfSamplesForDensityProfile,
                                      ActiveVoltage, D_T, D_R, ForceFactor, dist_Eq, velocity, bins, AverageForceLeft,
                                      AverageForceRight,
                                      SumAverageForce, ProbAtCenter, NumberOfParticles, ForceExp, f_max_N, y_wl, y_wr,
                                      ChannelLength_um,
                                      epsilon, radius, PressureLeft_NPum, PressureRight_NPum, SumPressure_NPum, BoxX,
                                      CurrentFolder, ppType,
                                      AverageForceLeft_lastHalf, AverageForceRight_lastHalf, SumAverageForce_lastHalf,
                                      PressureLeft_NPum_lastHalf, PressureRight_NPum_lastHalf, SumPressure_NPum_lastHalf, C, arm
                                      )
                        writeProbDistFunc(filenameBase, CurrentFolder, x_hist_lastHalf, hist_lastHalf)
                        if SaveCoords:
                            np.savetxt(subFolder + "/" + filenameBase + "_coords.csv", coords, delimiter=",")


# save force profile to folder
ExportArray=np.vstack((dist_Eq,Force_Eq))
np.savetxt(subFolder + "/"  + "UsedEq_ForceProfile.csv", ExportArray.T, delimiter=",")
print(subFolder + "/" + subFolder + "UsedEq_ForceProfile.csv")
