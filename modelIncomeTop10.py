#! /usr/bin/env python

#
# Script name: modelIncomeTop10.py
# Copyright 2021 Anonymous Author
#
# Version History:
#   Version 1.0:  Anonymous Review Release.  14 June 2021.
#
# License: see LICENSE.md

import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import matplotlib
import modelPID as mp


matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 

# main

# The first row is garbage
# The 0 column is "year from start"
# The 1 column is the equity gap
fname    = 'PicketySaezIncomeTopDecile.csv'
data_in  = np.loadtxt(fname, delimiter=',', skiprows=1, usecols=(0,1))
# Use data only post WW2.
year0ind = 28
N        = data_in[year0ind:, 0]  # This data set is in forward time order.
x        = data_in[year0ind:, 1] 
rows     = len(x)
#N        = N - year0
x        = (x / 10.0) - 1   # The ideal would be 10%, so the "inequality" is 10 (?)


# # Calc a model using some segment of the data and a setpoint=0
# T         = [0, 30, 60]
# k_est0    = mp.estPIDModel(x[0:], 0.0)
# k_est1    = mp.estPIDModel(x[15:], 0.0)
# k_est2    = mp.estPIDModel(x[30:], 0.0)


# # Simulate the estimated model for a specified period of time
# noiseStd = np.array([0.0, 0.0, 0.0])
# simDuration = 120
# [simOut0, timeSim0] = mp.simulatePIDModel(T[0], simDuration, k_est0, x)
# [simOut1, timeSim1] = mp.simulatePIDModel(T[1], simDuration-T[1], k_est1, x)
# [simOut2, timeSim2] = mp.simulatePIDModel(T[2], simDuration-T[2], k_est2, x)


# plt.ion()
# plt.figure(1)
# plt.clf()
# plt.plot(timeSim0 + year0, simOut0[:,0], label='Model Start '+str(year0+T[0]))
# plt.plot(timeSim1 + year0, simOut1[:,0], label='Model Start '+str(year0+T[1]))
# plt.plot(timeSim2 + year0, simOut2[:,0], label='Model Start '+str(year0+T[2]))
# plt.plot(N + year0, x, label='Actual')
# plt.xlim(year0, year0 + simDuration-1)
# plt.xlabel("Year", fontsize=20)
# plt.ylabel("Top 10% Income Percentage - 10%", fontsize=20)
# plt.xticks(range(year0, year0+simDuration, 10))
# plt.legend(fontsize=16)
# plt.grid(True)
# plt.show()




plt.ion()

# Calc a model using some segment of the data and a setpoint=0
plt.figure(1)
trainingRows = int(rows/2.0)
# extrapolateViaModel(x, N, trainingRows=0, setpoint=0.0, ylabelstr='Inequality Ratio', tickDelta=8)
mp.extrapolateViaModel(x, N, trainingRows, 0.0, "Total Income of Top 10% / 10%", 10)
plt.legend(loc='upper left', fontsize=14)
plt.ylim(0.95, 6)
plt.show()
plt.savefig("extrapolate_incomeTop10_1982-2018_kalman_ratio.eps")
plt.savefig("extrapolate_incomeTop10_1982-2018_kalman_ratio.png")

# Calc a model using some segment of the data and a setpoint=0
print()
plt.figure(2)
trainingRows = int(2.0*rows/3.0)
# extrapolateViaModel(x, N, trainingRows=0, setpoint=0.0, ylabelstr='Inequality Ratio', tickDelta=8)
mp.extrapolateViaModel(x, N, trainingRows, 0.0, "Total Income of Top 10% / 10%", 10)
plt.legend(loc='upper left', fontsize=14)
plt.ylim(0.95, 6)
plt.show()
plt.savefig("extrapolate_incomeTop10_1994-2018_kalman_ratio.eps")
plt.savefig("extrapolate_incomeTop10_1994-2018_kalman_ratio.png")

# Estimate parameters from all data
print()
k_training   = mp.estimatePIDModelKalman(x, 0.0)

# Estimate parameters from 2nd half of data
print()
k_training   = mp.estimatePIDModelKalman(x[int(rows/2.0):], 0.0)

print()
# Calc a model using some segment of the data and a setpoint=0
plt.figure(3)
futureTimes = 20
mp.extrapolateViaModelFuture(x, N, futureTimes, 0.0, "Total Income of Top 10% / 10%", 15)
plt.legend(loc='lower left', fontsize=14)
plt.ylim(0.95, 6)
plt.show()
plt.savefig("extrapolate_incomeTop10_future_kalman_ratio.eps")
plt.savefig("extrapolate_incomeTop10_future_kalman_ratio.png")

