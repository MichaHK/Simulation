import numpy as np
Range_um_for_OrderParameterCalc = 2
LocationsToReportOrderParameter = np.array([9, 50 , 89])



y = 52
p = 0.05


S = np.zeros([3,1])
SampleCounters = np.zeros([3,1])
BoolForSampleYs = (LocationsToReportOrderParameter - y)**2 < Range_um_for_OrderParameterCalc**2
S[BoolForSampleYs] = + p
SampleCounters[BoolForSampleYs] = + 1
print(S, SampleCounters)
# if coordsForRunning[jn, particle * 3 + 2]
