#! /usr/bin/env python

#
# Script name: predictFutureeWithDifferentKparams.py
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
futureRows = 15  # How many time steps into the future to predict.
plt.ion()


# Use Voting Data
fname    = 'BlackWhiteVotingPercentageUS.csv'
data_in  = np.loadtxt(fname, delimiter=',', skiprows=1, usecols=(0,1,2,3))
rows     = data_in.shape[0]
# Keep track of the earliest year; make it year 0; each row is two years.
year0    = data_in[-1,0]    # Last year in the data set
N        = np.flip(data_in[0:, 0]) 
x        = np.flip(data_in[0:, 2] / data_in[0:,3]) - 1.0


# Estimate the parameters of the model with 1/2 of the data as training.
trainingRows = int(rows/2.0)
setpoint     = 0
k_training   = mp.estimatePIDModelKalman(x[0:trainingRows], setpoint)
xhat         = mp.estimateEndingState(k_training, x[0:trainingRows])

# Simulate the estimated model for a specified period of time
simDuration  = rows-trainingRows
simOut_new = [[],[],[],[],[],[]]
simOut_new2 = [[],[],[],[],[],[]]
# Simulate using estimated k parameters
[simOut_orig, timeSim] = mp.simulatePIDModel(trainingRows, simDuration, k_training, x, \
		xhat[0], xhat[1], xhat[2])
# Simulate with each parameter cut in half
print("----")
print(k_training)
for param in range(3):
	k_new = k_training.copy()
	if param==1:
		k_new[param] = k_training[param]/2.0
	else:
		k_new[param] = k_training[param]*2.0
	print(k_new)
	[simOut_new[param], timeSim] = mp.simulatePIDModel(trainingRows, simDuration, k_new, x, \
		xhat[0], xhat[1], xhat[2])



plt.ion()
plt.figure(1)
plt.plot(N[0:trainingRows], x[0:trainingRows] + 1.0, 'g-o', linewidth=2, label='Actual Training')
plt.plot(N[trainingRows:], simOut_orig[:,0] + 1.0, 'b-', linewidth=1, label='Original k')
plt.plot(N[trainingRows:], simOut_new[0][:,0] + 1.0, '-', color="tab:orange", linewidth=1, label='With k_P*2')
plt.plot(N[trainingRows:], simOut_new[2][:,0] + 1.0, '-', color="tab:green", linewidth=1, label='With k_I*2')
plt.plot(N[trainingRows:], simOut_new[1][:,0] + 1.0, '-', color="tab:red", linewidth=1, label='With k_D/2')
plt.legend(loc='lower left', fontsize=14)
plt.ylim(0.9, 1.6)
plt.plot([min(N)-0.5, max(N)+0.5], [1, 1], 'k-', linewidth=1)
plt.xlim(min(N)-0.5, max(N)+0.5)
plt.xlabel("Year", fontsize=20)
plt.ylabel("White / Black Voting %", fontsize=20)
plt.xticks(range(int(min(N)), int(max(N)), 8))
plt.legend(fontsize=16)
plt.grid(True)
plt.show()
plt.savefig("predictFutureeWithDifferentKparams_Voting.eps")
plt.savefig("predictFutureeWithDifferentKparams_Voting.png")

