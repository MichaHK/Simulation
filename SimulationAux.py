import numpy as np
import pandas as pd
from forces import *
import datetime


def NoReason():
    print('s')


def CalcPDFForYlocations(y, bins=400):
    hist, bin_edges = np.histogram(y, density=True, bins=bins)
    x_hist = bin_edges[0] + np.cumsum(np.diff(bin_edges)) - np.diff(bin_edges).mean() / 2
    return x_hist, hist


def writeProbDistFunc(filenameBase, CurrentFolder, x_hist, hist):
    ExportFilename3 = CurrentFolder + '/' + filenameBase + '_PDfunc.csv'
    (pd.DataFrame(data=np.column_stack([x_hist, hist]), columns=['x_um', 'pdf'])).to_csv(ExportFilename3,
                                                                                         index_label=False, index=False)


def writeMetaData(filenameBase, kT,drag_N_sPum,drag_multiplier,dt,numberOfSteps,NumberOfSamplesForDensityProfile,
                  ActiveVoltage, D_T, D_R, ForceFactor, dist_Eq, velocity, bins, AverageForceLeft,AverageForceRight,
                  SumAverageForce, ProbAtCenter, NumberOfParticles,ForceExp ,f_max_N ,y_wl ,y_wr ,ChannelLength_um,
                  epsilon, radius,PressureLeft_NPum ,PressureRight_NPum, SumPressure_NPum, BoxX, CurrentFolder, ppType,
                  AverageForceLeft_lastHalf, AverageForceRight_lastHalf, SumAverageForce_lastHalf,
                  PressureLeft_NPum_lastHalf, PressureRight_NPum_lastHalf, SumPressure_NPum_lastHalf, C, arm,
                  SamplingTimeForDensityProfile):
    NowDateObj = datetime.datetime.now()
    comments = "Force profile is saved to folder, so just multiply by ForceFactor to get the profile used here"
    exportDict = {'kT_J': kT, 'drag_N_sPum': drag_N_sPum, 'drag_multiplier': drag_multiplier,
                  'dt_s': dt,
                  'numberOfSteps': numberOfSteps,
                  'dt_betweenProfilePoints_s': dt * numberOfSteps / NumberOfSamplesForDensityProfile,
                  "D_T_umPs": D_T, "D_R_Ps": D_R, "ActiveVoltage": ActiveVoltage,
                  "ForceFactor": ForceFactor,
                  "x_min_um": dist_Eq.min(),
                  "x_max_um": dist_Eq.max(), "velocity_umPs": velocity, 'bins': bins,
                  'AverageForceLeft_N': AverageForceLeft, 'AverageForceRight_N': AverageForceRight,
                  'SumAverageForce_N': SumAverageForce, 'filenameBase': filenameBase,
                  "ProbAtCenter": ProbAtCenter, "NumberOfParticles": NumberOfParticles,
                  "ForceExp": ForceExp, "f_max_N": f_max_N, "y_wl": y_wl, "y_wr": y_wr,
                  "ChannelLength_um": ChannelLength_um, "epsilon": epsilon, "radius": radius,
                  "PressureLeft_NPum": PressureLeft_NPum, "PressureRight_NPum": PressureRight_NPum,
                  "SumPressure_NPum": SumPressure_NPum, "BoxX_um": BoxX, "ppType": ppType,
                  "PressureLeft_NPum_lastHalf": PressureLeft_NPum_lastHalf, "PressureRight_NPum_lastHalf": PressureRight_NPum_lastHalf, "SumPressure_NPum_lastHalf": SumPressure_NPum_lastHalf,
                  'AverageForceLeft_N_lastHalf': AverageForceLeft_lastHalf, 'AverageForceRight_N_lastHalf': AverageForceRight_lastHalf, 'SumAverageForce_N_lastHalf': SumAverageForce_lastHalf,
                  "C_ratio_dlctrc_cnst": C, "arm_um_COM_to_HemCntr": arm,
                  "comments": comments, "SamplingTimeForCoordsFile_s": SamplingTimeForDensityProfile
                  }  # "polynomParams": popt, 'PolynomCode': str(inspect.getsourcelines(ForcePolynomSym)[0]).replace(',',' '),
    ExportFilename2 = CurrentFolder + '/' + filenameBase + '_MetaData.csv'
    # if not os.path.isfile(fname): # if doesn't exist write headers too.
    with open(ExportFilename2, 'w') as file:
        for x in exportDict.keys():
            file.write(str(x) + ',')
        file.write('\n')
    with open(ExportFilename2, 'a') as file:
        for x in exportDict.values():
            file.write(str(x) + ',')
        file.write('\n')
    print(ExportFilename2)


def getFilenameBase(ActiveVoltage, velocity, D_T, D_R, NumberOfParticles, drag_multiplier, C):
    string = ('Sim_V_' + str(ActiveVoltage).replace('.', 'p') +
              '_vel_' + str(velocity).replace('.', 'p') +
              '_Dt_' + str(D_T).replace('.', 'p') +
              '_C_' + str(C).replace('.', 'p') +
              '_Dr_' + str(D_R).replace('.', 'p') +
              '_NumPrtcls_' + str(NumberOfParticles) +
              '_dragFactorToPsv_' + str(np.round(drag_multiplier, 2)).replace('.', 'p'))
    return string


def getAverageForceAsym(x_hist, hist, ForceFactor,
                        f_max_N, y_wl, y_wr, ChannelLength_um, WallType = 'linear'):
    dx = np.diff(x_hist).mean()
    ForcesAvg = np.empty(len(hist))
    AverageForceLeft = 0
    AverageForceRight = 0
    ind = 0
    for x, pdf in zip(x_hist, hist):
        if WallType == 'linear':
            ForcesAvg[ind] = wall_force_linear(x, f_max_N, y_wl, y_wr, ChannelLength_um) * pdf * ForceFactor * dx
        if WallType == 'BiHarm':
            ForcesAvg[ind] = wall_force_biharm(x, f_max_N, y_wl, y_wr, ChannelLength_um) * pdf * ForceFactor * dx
        if WallType == 'experimental':
            ForcesAvg[ind] = exp_wall_force(x, ChannelLength_um) * pdf * ForceFactor * dx
        if ForcesAvg[ind] > 0:
            AverageForceLeft += ForcesAvg[ind]
        elif ForcesAvg[ind] < 0:
            AverageForceRight += ForcesAvg[ind]
        ind += 1
    # ForcesAvg = np.array(ForcesAvg)
    # AverageForceLeft = np.sum(ForcesAvg[x_hist < y_wl])
    # AverageForceRight = np.sum(ForcesAvg[x_hist > y_wr])
    SumAverageForce = AverageForceRight + AverageForceLeft
    return AverageForceLeft, AverageForceRight, SumAverageForce
