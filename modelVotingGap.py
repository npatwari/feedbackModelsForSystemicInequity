#! /usr/bin/env python

#
# Script name: modelVotingGap.py
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


matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 

# main

# The first row is garbage
# The 0 column is "year from start"
# The 1 column is the equity gap
fname    = 'BlackWhiteVotingPercentageUS.csv'
data_in  = np.loadtxt(fname, delimiter=',', skiprows=1, usecols=(0,1,2,3))
rows     = data_in.shape[0]

# Keep track of the earliest year; make it year 0; each row is two years.
year0    = data_in[-1,0]    # Last year in the data set

# N is what is used to plot against on the y-axis.
# x is the data being predicted.
# Flip is because I know this data set is in reverse time order.
N        = np.flip(data_in[0:, 0]) 
#x        = np.flip(data_in[0:, 1]) 

# Ratio minus 1 (equality)
x        = np.flip(data_in[0:, 2] / data_in[0:,3]) - 1.0

# Calc a model using some segment of the data and a setpoint=0
# T         = [0, 8, 15]
# k_est0    = mp.estPIDModel(x[T[0]:], 0.0)
# k_est1    = mp.estPIDModel(x[T[1]:], 0.0)
# k_est2    = mp.estPIDModel(x[T[2]:], 0.0)


# Simulate the estimated model for a specified period of time
# noiseStd = np.array([0.0, 0.0, 0.0])
# simDuration = 55
# [simOut0, timeSim0] = mp.simulatePIDModel(T[0], simDuration, k_est0, x)
# [simOut1, timeSim1] = mp.simulatePIDModel(T[1], simDuration-T[1], k_est1, x)
# [simOut2, timeSim2] = mp.simulatePIDModel(T[2], simDuration-T[2], k_est2, x)


# plt.ion()
# plt.figure(1)
# plt.clf()
# plt.plot(timeSim0*2 + year0, simOut0[:,0], label='Model Start '+str(int(year0+T[0]*2)))
# plt.plot(timeSim1*2 + year0, simOut1[:,0], label='Model Start '+str(int(year0+T[1]*2)))
# plt.plot(timeSim2*2 + year0, simOut2[:,0], label='Model Start '+str(int(year0+T[2]*2)))
# plt.plot(N*2 + year0, x, label='Actual')
# plt.xlabel("Year", fontsize=20)
# plt.ylabel("White Voting % - Black Voting % ", fontsize=20)
# plt.xticks(range(int(year0), int(year0+simDuration*2), 20))
# plt.legend(fontsize=16)
# plt.grid(True)
# plt.show()




plt.ion()

# Calc a model using some segment of the data and a setpoint=0
plt.figure(1)
trainingRows = int(rows/2.0)
# extrapolateViaModel(x, N, trainingRows=0, setpoint=0.0, ylabelstr='Inequality Ratio', tickDelta=8)
mp.extrapolateViaModel(x, N, trainingRows, 0.0, "White / Black Voting %", 8)
plt.legend(loc='lower left', fontsize=14)
plt.ylim(0.8, 1.5)
plt.show()
plt.savefig("extrapolate_votinggap_1992-2018_kalman_ratio.eps")
plt.savefig("extrapolate_votinggap_1992-2018_kalman_ratio.png")

print()
# Calc a model using some segment of the data and a setpoint=0
plt.figure(2)
trainingRows = int(2.0*rows/3.0)
# extrapolateViaModel(x, N, trainingRows=0, setpoint=0.0, ylabelstr='Inequality Ratio', tickDelta=8)
mp.extrapolateViaModel(x, N, trainingRows, 0.0, "White / Black Voting %", 8)
plt.legend(loc='lower left', fontsize=14)
plt.ylim(0.8, 1.5)
plt.show()
plt.savefig("extrapolate_votinggap_2000-2018_kalman_ratio.eps")
plt.savefig("extrapolate_votinggap_2000-2018_kalman_ratio.png")

# Estimate parameters from all data
print()
k_training   = mp.estimatePIDModelKalman(x, 0.0)


# Estimate parameters from 2nd half of data
print()
k_training   = mp.estimatePIDModelKalman(x[int(rows/2.0):], 0.0)

print()
# Calc a model using some segment of the data and a setpoint=0
plt.figure(3)
futureTimes = 10
mp.extrapolateViaModelFuture(x, N, futureTimes, 0.0, "White / Black Voting %", 12)
plt.legend(loc='upper right', fontsize=14)
plt.ylim(0.8, 1.5)
plt.show()
plt.savefig("extrapolate_votinggap_future_kalman_ratio.eps")
plt.savefig("extrapolate_votinggap_future_kalman_ratio.png")

