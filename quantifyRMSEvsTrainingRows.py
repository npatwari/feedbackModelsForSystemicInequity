#! /usr/bin/env python

#
# Script name: quantifyRMSEvsTrainingRows.py
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


# Use gender pay gap data
fname    = 'GenderPayGapData.csv'
data_in  = np.loadtxt(fname, delimiter=',', skiprows=2, usecols=(0,1))
rows     = data_in.shape[0]
N        = np.flip(data_in[:, 0])  + 1960
x        = 1.0/(1.0 - np.flip(data_in[0:, 1]) ) - 1.0
# Calc the RMSE for each size of the training.
for trainingRows in range(trainingMin, rows-futureRows, 1):
	simOut_test, err, rmse, mae = mp.extrapolateViaModel(x[0:(trainingRows+futureRows)], N[0:(trainingRows+futureRows)], trainingRows, 0.0, "White / Black Voting %", 8)
	averageValue = np.mean(x[0:(trainingRows+futureRows)])
	rmse_save[0].append(rmse/averageValue)
	mae_save[0].append(mae/averageValue)
	trainingRows_save[0].append(trainingRows)



# Use Voting Data
fname    = 'BlackWhiteVotingPercentageUS.csv'
data_in  = np.loadtxt(fname, delimiter=',', skiprows=1, usecols=(0,1,2,3))
rows     = data_in.shape[0]
# Keep track of the earliest year; make it year 0; each row is two years.
year0    = data_in[-1,0]    # Last year in the data set
N        = np.flip(data_in[0:, 0]) 
x        = np.flip(data_in[0:, 2] / data_in[0:,3]) - 1.0

# Calc the RMSE for each size of the training.
for trainingRows in range(trainingMin, rows-futureRows, 1):
	simOut_test, err, rmse, mae = mp.extrapolateViaModel(x[0:(trainingRows+futureRows)], N[0:(trainingRows+futureRows)], trainingRows, 0.0, "White / Black Voting %", 8)
	averageValue = np.mean(x[0:(trainingRows+futureRows)])
	rmse_save[1].append(rmse/averageValue)
	mae_save[1].append(mae/averageValue)
	trainingRows_save[1].append(trainingRows)



# Use Income Data
fname    = 'PicketySaezIncomeTopDecile.csv'
data_in  = np.loadtxt(fname, delimiter=',', skiprows=1, usecols=(0,1))
# Use data only post WW2.
year0ind = 28
N        = data_in[year0ind:, 0]  # This data set is in forward time order.
x        = data_in[year0ind:, 1] 
rows     = len(x)
x        = (x / 10.0) - 1   # The ideal would be 10%, so the "inequality" is 10 (?)

# Calc the RMSE for each size of the training.
for trainingRows in range(trainingMin, rows-futureRows, 1):
	simOut_test, err, rmse, mae = mp.extrapolateViaModel(x[0:(trainingRows+futureRows)], N[0:(trainingRows+futureRows)], trainingRows, 0.0, "White / Black Voting %", 8)
	averageValue = np.mean(x[0:(trainingRows+futureRows)])
	rmse_save[2].append(rmse/averageValue)
	mae_save[2].append(mae/averageValue)
	trainingRows_save[2].append(trainingRows)


with open('quantifyRMSEvsTrainingRows.pkl') as f:  # Python 3: open(..., 'rb')
    trainingRows_save, mae_save, rmse_save = pickle.load(f)


# Plot the result
plt.close(9)
plt.figure(9)
for setnum in range(3):
	plt.plot(trainingRows_save[setnum], rmse_save[setnum], label="Dataset "+str(setnum))
plt.legend(fontsize=16)


# Plot the result
plt.close(10)
plt.figure(10)
markerlist = ['bo', 'gx', 'r*']
for setnum in range(3):
	plt.plot(trainingRows_save[setnum], mae_save[setnum], markerlist[setnum], label="Dataset "+str(setnum+1))
plt.legend(fontsize=16)
plt.ylim(0, 1.8)
plt.yticks(np.arange(0, 1.801, 0.3))
plt.xlim(4, 63)
plt.grid("on")
plt.xlabel('Training Set Data Length', fontsize=16)
plt.ylabel('Avg Normalized Extrapolation Error', fontsize=16)


with open('quantifyRMSEvsTrainingRows.pkl', 'w') as f:  # Python 3: open(..., 'wb')
    pickle.dump([trainingRows_save, mae_save, rmse_save], f)
