import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import ceil,sqrt,floor

def DrawParticles(img,particleLocations,radius,ax):
    fig, ax = plt.subplots(1,1)
    for i in particleLocations:
        # draw the outer circle
        ax.plot(i[0],i[1],markersize=4)

def GetInitialPts(NumberOfParticles,BoxX,BoxY):
    x_pos_lattice=np.linspace(0.05,0.95,int(ceil(sqrt(NumberOfParticles*BoxX/BoxY))))*BoxX
    # x_pos_lattice=np.concatenate([x_pos_lattice,x_pos_lattice+2*radius],axis=0) # for test, removes later
    y_pos_lattice=np.linspace(0.1,0.9,int(ceil(sqrt(NumberOfParticles*BoxY/BoxX))))*BoxY
    # y_pos_lattice=np.concatenate([y_pos_lattice,y_pos_lattice],axis=0) # for test, removes later
    pts=list()
    for y in y_pos_lattice:
        for x in x_pos_lattice:
            pts.append((x,y))
    InitialPts=pts[len(pts)-NumberOfParticles:]
    return InitialPts
