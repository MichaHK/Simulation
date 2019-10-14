def main():
    pass

if __name__ == "__main__":
    # load imported
    # import trackpy as tp
    import matplotlib.pyplot as plt
    import time
    import os
    import numpy as np
    from forces import *
    from SimulationAux import writeProbDistFunc, CalcPDFForYlocations,writeMetaData, getFilenameBase, getAverageForceAsym
    from PlaceInitialParticles import *
    from SimulationLoop import runSimulationPPInt
    from math import ceil,floor,sqrt


    plt.rc('text', usetex=True)
    plt.rcParams['font.size']=20
    plt.rcParams["font.family"] = "Times New Roman"


    ### Setting pp interactions
    epsilon=10**(-20)
    radius=1
    ###  setting wall interactions
    Voltage_eq=1;
    ForceExp=1; # change it here or meta files will be bad.
    f_max_N=5*10**(-15)
    y_wl=20;
    y_wr=80;
    ChannelLength_um=100
    BoxX=ChannelLength_um*2;
    ### setting simulation params
    numberOfSteps=10000000; # USE it
    ActiveVoltages=[7];
    D_T= 0.1 # use um^2/s, translational
    D_Rs=[1/10]
    NumberOfParticlesArray=[400]
    RhoZeroLocation_um=50
    velocities=[2]
    drag_multiplier=1
    kT=4.11*10**(-21)
    bins=400
    dt=0.00001


    ### init pp interactions
    IntRange=2*radius*2**(1/6)

    ### init wall interactions
    dist_Eq=np.linspace(0,ChannelLength_um,ChannelLength_um+1)
    BoxY=ChannelLength_um; # else the np.mod in brwonian motion would not work)
    Force_Eq=np.array([wall_force_linear(y,f_max_N,y_wl,y_wr,ChannelLength_um) for y in dist_Eq])

    ### folder to save
    subFolder='SimulationsNoTorque'
    CurrentFolder=os.getcwd()
    CurrentFolder=CurrentFolder.replace('\\',r'/')+'/'+subFolder
    CurrentFolder=CurrentFolder.replace('/notebooks','')


    # run the simulation
    drag_N_sPum=drag_multiplier*kT/D_T*10**6
    for D_R in D_Rs:
        for ActiveVoltage in ActiveVoltages:
            ForceFactor=1*(ActiveVoltage/Voltage_eq)**2
            for velocity in velocities:
                if velocity<Force_Eq.max()*ForceFactor/drag_N_sPum:
                    for NumberOfParticles in NumberOfParticlesArray:
                        print('run...')
                        SamplingTimeForDensityProfile=radius/(velocity/D_R)/D_R
                        NumberOfSamplesForDensityProfile=ceil(numberOfSteps*dt/SamplingTimeForDensityProfile)
                        coordsForProfile=np.empty([NumberOfSamplesForDensityProfile,3*NumberOfParticles]) # p1x1y1 (particle 1), p2x2y2 (particle 2),.... [array] p is the angle. Each row is a timestep.
                        coordsForRunning=np.empty([2,3*NumberOfParticles]) # p1x1y1 (particle 1), p2x2y2 (particle 2),.... [array] p is the angle. Each row is a timestep.
                        # setting initial conditions: positions, velocities.
                        InitialPts=GetInitialPts(NumberOfParticles,BoxX,BoxY)
                        coordsForRunning[0,1::3]=np.array(InitialPts)[:,0]; # inital x coordinates.
                        coordsForRunning[0,2::3]=np.array(InitialPts)[:,1]; # inital y coordinates.
                        coordsForRunning[0,0::3]=sqrt(2*D_R*dt) * np.random.randn(len(InitialPts)) + 0
                        # run the loops
                        t=time.time()
                        coords=runSimulationPPInt(coordsForProfile, coordsForRunning, velocity, NumberOfParticles,
                                                                     dt, numberOfSteps, NumberOfSamplesForDensityProfile,
                                                                     drag_N_sPum,ForceFactor, ChannelLength_um,
                                                  radius, BoxX, BoxY, IntRange,D_T,D_R,y_wl,y_wr,f_max_N,epsilon)
                        elapsed=time.time()-t;
                        print(elapsed)

                        # exporting
                        x_hist,hist =CalcPDFForYlocations(coords[:,2::3],bins=bins)
                        indToRhoZero=np.argmin(np.abs(x_hist-RhoZeroLocation_um))
                        ProbAtCenter=hist[indToRhoZero]
                        # plt.plot(x_hist,hist)
                        # calculate pressure
                        AverageForceLeft,AverageForceRight,SumAverageForce=getAverageForceAsym(x_hist,hist,ForceFactor,
                                                                                               f_max_N,y_wl,y_wr,ChannelLength_um)
                        PressureLeft_NPum,PressureRight_NPum,SumPressure_NPum=np.array(
                            [AverageForceLeft,AverageForceRight,SumAverageForce])*NumberOfParticles/BoxX
                        # print(AverageForceLeft,AverageForceRight,SumAverageForce)
                        print(PressureLeft_NPum,PressureRight_NPum,SumPressure_NPum)
                        # export
                        filenameBase=getFilenameBase(ActiveVoltage,velocity,D_T,D_R,NumberOfParticles,drag_multiplier);
                        writeMetaData(filenameBase, kT,drag_N_sPum,drag_multiplier,dt,numberOfSteps,NumberOfSamplesForDensityProfile,
                                          ActiveVoltage, D_T, D_R, ForceFactor, dist_Eq, velocity, bins, AverageForceLeft,AverageForceRight,
                                          SumAverageForce, ProbAtCenter, NumberOfParticles,ForceExp ,f_max_N ,y_wl ,y_wr ,ChannelLength_um,
                                          epsilon, radius,PressureLeft_NPum ,PressureRight_NPum, SumPressure_NPum, BoxX, CurrentFolder)
                        writeProbDistFunc(filenameBase,CurrentFolder,x_hist,hist)
    main()
