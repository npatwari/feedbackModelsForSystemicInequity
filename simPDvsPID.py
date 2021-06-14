#! /usr/bin/env python

#
# Script name: simPDvsPID.py
# Copyright 2021 Anonymous Author
#
# Version History:
#   Version 1.0:  Anonymous Review Release.  14 June 2021.
#
# License: see LICENSE.md

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import modelPID as mp
import pickle

matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 

# main
rmse_save = [[], [], []]
mae_save = [[], [], []]
trainingRows_save = [[], [], []]
trainingMin = 5
futureRows = 10  # How many time steps into the future to predict.
plt.ion()

startYear = 0
duration = 26
k = [-0.3, -0.8, -0.05]
init_S = [1, 0, 0]
noiseMean = 0.1
noiseStd = 0.001
simOutPID, timeSim = mp.simulatePIDModelInNoise(startYear, duration, k, init_S, noiseMean, noiseStd)
k = [-0.3, -0.8, 0.0]
init_S = [1, 0, 0]
simOutPD, timeSim = mp.simulatePIDModelInNoise(startYear, duration, k, init_S, noiseMean, noiseStd)


# Plot the result
plt.close(11)
plt.figure(11)
plt.plot(timeSim, simOutPD[:,0]+1, 'k-', linewidth=2, label="PD Feedback")
plt.plot(timeSim, simOutPID[:,0]+1, 'k:', linewidth=2, label="PID Feedback")
plt.legend(fontsize=16)
plt.xlim(1, 25)
plt.grid("on")
plt.xlabel('Time (Years)', fontsize=16)
plt.ylabel('Inequity Ratio', fontsize=16)
plt.show()


filename = "simPDvsPID_" + str(k[0]) + "_kI" + str(k[2]) + "_kD" + str(k[1]) + "_mu"+ str(noiseMean) + "_sigma" + str(noiseStd) 

plt.savefig(filename + ".eps")
plt.savefig(filename + ".png")



