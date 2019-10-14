# try to

### load imported
import time
import os
import numpy as np
from forces import *
from SimulationAux import writeProbDistFunc, CalcPDFForYlocations, writeMetaData, getFilenameBase, \
    getAverageForceAsym
from PlaceInitialParticles import *
from SimulationLoop import runSimulationPPInt, runSimulationTorqueNoPPInt
from math import ceil, floor, sqrt
import matplotlib.pyplot as plt

subFolder = 'SimulationsWithTorque'
### Setting pp interactions
epsilon = 10 ** (-15)
radius = 1
ppType = 'WCA'
###  setting wall interactions
Voltage_eq = 1
ForceExp = 1  # change it here or meta files will be bad.
f_max_N = 5 * 10 ** (-15)
y_wl = 25
y_wr = 75
RhoZeroLocation_um = np.mean([y_wl, y_wr])
ChannelLength_um = 100
BoxX = ChannelLength_um * 2
### setting simulation params
ActiveVoltages = [7]
D_T = 0.1  # use um^2/s, translational
D_Rs = [1 / 10]
NumberOfParticlesArray = [100, 500,  1000, 2000, 5000]
dts = [0.005, 0.005, 0.001, 0.0004, 0.0004]
numberOfSteps_s = [3000000, 1000000, 1000000, 1000000, 800000]  # USE it 1500000

velocities = [2,3,4]
drag_multiplier = 1
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
Force_Eq = np.array([wall_force_linear(y, f_max_N, y_wl, y_wr, ChannelLength_um) for y in dist_Eq])
# Force_Eq = np.array([wall_force_biharm(y, f_max_N, y_wl, y_wr, ChannelLength_um) for y in dist_Eq])

# Force_Eq= np.array([exp_wall_force(y, ChannelLength_um) for y in dist_Eq])

plt.interactive(False)
plt.plot(dist_Eq, Force_Eq)
plt.xlabel('$d$ [$\mu$m]', fontsize=20)
plt.ylabel('$f_{\mathrm{wall}}$ [N]', fontsize=20)
plt.tight_layout()
plt.show()
