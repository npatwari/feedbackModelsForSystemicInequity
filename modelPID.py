#! /usr/bin/env python

#
# Script name: modelPID.py
# Copyright 2021 Anonymous Author
#
# Version History:
#   Version 1.0:  Anonymous Review Release.  14 June 2021.
#
# License: see LICENSE.md

import numpy as np
import numpy.linalg as linalg
# https://pykalman.github.io/
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 



# Estimate the three parameters of the PID model for the system,
# given the data from the progression of the system, a setpoint, 
# and assuming that the three state components are the inequity,
# the 1-year slope of inequity, and the cumulative inequity (since year 0).
def estPIDModel(x, setpoint=0):

	rows = x.shape[0]
	xp = x - setpoint         # The inequity at each time
	# The slope of the inequity xdot(n) = x(n)-x(n-1).
	# Note the 0 element of xdot is really at time 1
	xdot = xp[1:] - xp[0:-1]    

	# The cumulative inequity since time 0.
	# Note the 0 element of xdot is really at time 0
	xsum = np.cumsum(xp)

	# The change in slope. Note the 0 element of xdotdot is really at time 1
	xdotdot = xdot[1:] - xdot[0:-1] # xdotdot(n) = x(n+1)-x(n)

	# We only really have rows-2 data points, we can't estimate xdot, xdotdot 
	# of the first year or xdotdot of the last year.
	S = np.zeros([rows-2, 3])
	S[:,0] = xp[1:-1]
	S[:,1] = xdot[0:-1]
	S[:,2] = xsum[1:-1]
	pseudoinverseS = linalg.pinv(S)

	model = pseudoinverseS.dot(xdotdot)
	return model

def estimateEndingState(k_est, x):

	# A is the linear model for how the state progresses from year to year.
	A         = np.array([[1, 1, 0], [k_est[0], k_est[1]+1, k_est[2]], [1, 1, 1]])

	# C is the observation matrix
	C = np.array([1, 0, 0])	

    # init Kalman filter and use it to estimate
	kf= KalmanFilter(A, C)

	kf.em(x, n_iter=5)
	(filtered_state_means, filtered_state_covs) = kf.filter(x)

	return filtered_state_means[-1,:]

def estimatePIDModelKalman(x, setpoint=0):

	k_est = estPIDModel(x, setpoint)
	print(k_est)

	for i in range(9):

		# A is the linear model for how the state progresses from year to year.
		A         = np.array([[1, 1, 0], [k_est[0], k_est[1]+1, k_est[2]], [1, 1, 1]])

		# C is the observation matrix
		C = np.array([1, 0, 0])	
		# init Kalman filter and use it to estimate
		kf= KalmanFilter(A, C)

		kf.em(x, n_iter=5)
		(smoothed_state_means, smoothed_state_covs) = kf.smooth(x)

		k_est = estPIDModel(smoothed_state_means[:,0], setpoint)
		print(k_est)

	return k_est






# Simulate the estimated model for a specified period of time
def simulatePIDModel(startYear, duration, k_est, x, init_x=None, init_slope=0, init_sum=0):

    # Initialize the first year of the simulation.  You could make different assumptions
    # about how the first simulation year should be initialized.  But we don't
    # have any data before year 0, so there might need to be a different method
    # for a start year of 0.
	if init_x is None:
	    if startYear >= 0 and startYear < len(x):
	        init_x = x[startYear]
	    else:
	    	init_x = 0

	init_S = np.array([init_x, init_slope, init_sum])  # Ensure it has three elements and is a np.array
	#print(init_S)

	# A is the linear model for how the state progresses from year to year.
	A         = np.array([[1, 1, 0], [k_est[0], k_est[1]+1, k_est[2]], [1, 1, 1]])
	
	# initialize the simulation output
	simOut    = np.zeros((duration,3))  
	simOut[0] = init_S

	# Calculate the next year's state from the prior year
	for i in range(duration-1):
		simOut[i+1] = A.dot(simOut[i])

    # Also return an array of the year numbers
	timeSim   = np.arange(startYear, startYear + duration)
	return simOut, timeSim


# Simulate the estimated model for a specified period of time
def simulatePIDModelInNoise(startYear, duration, k_est, init_S, noiseMean, noiseStd):


	# A is the linear model for how the state progresses from year to year.
	A         = np.array([[1, 1, 0], [k_est[0], k_est[1]+1, k_est[2]], [1, 1, 1]])
	
	# initialize the simulation output
	simOut    = np.zeros((duration,3))  
	simOut[0] = init_S


	# Calculate the next year's state from the prior year
	for i in range(duration-1):
		simOut[i+1] = A.dot(simOut[i]) 

		# Add in Gaussian noise with givem mean and std, to the proportional term.
		noise = np.random.normal(noiseMean, noiseStd)
		simOut[i+1][1] += noise  

    # Also return an array of the year numbers
	timeSim   = np.arange(startYear, startYear + duration)
	return simOut, timeSim



def extrapolateViaModel(x, N, trainingRows=0, setpoint=0.0, ylabelstr='Inequality Ratio', tickDelta=8):
	rows         = len(x)
	if (trainingRows <=0) or (trainingRows >= (rows-1)):
		trainingRows = int(rows/2)

	k_training   = estimatePIDModelKalman(x[0:trainingRows], setpoint)

	# Simulate the estimated model for a specified period of time
	simDuration  = rows-trainingRows

	xhat         = estimateEndingState(k_training, x[0:trainingRows])

	# initalize the slope with the average slope over the first half.
	[simOut_test, timeSim0] = simulatePIDModel(trainingRows, simDuration, k_training, x, \
		xhat[0], xhat[1], xhat[2])

    #Calculate RMSE
	err = np.square(simOut_test[:,0] - x[trainingRows:])
	mae = np.sum(np.sqrt(err))/len(err)
	rmse = np.sqrt(np.sum(err)/len(err))
	print("RMSE = " + str(rmse))

	#plt.clf()
	plt.plot(N[0:trainingRows], x[0:trainingRows] + 1.0, 'g-o', linewidth=2, label='Actual Training')
	plt.plot(N[trainingRows:], simOut_test[:,0] + 1.0, 'b-', linewidth=1, label='Predicted Test')
	plt.plot(N[trainingRows:], x[trainingRows:] + 1.0, 'g:', linewidth=2, label='Actual Test')
	plt.plot([min(N)-0.5, max(N)+0.5], [1, 1], 'k-', linewidth=1)
	plt.xlim(min(N)-0.5, max(N)+0.5)
	plt.xlabel("Year", fontsize=20)
	plt.ylabel(ylabelstr, fontsize=20)
	plt.xticks(range(int(min(N)), int(max(N)), tickDelta))
	plt.legend(fontsize=16)
	plt.grid(True)
	plt.show()

	return simOut_test, err, rmse, mae

def extrapolateViaModelFuture(x, N, futureRows=0, setpoint=0.0, ylabelstr='Inequality Ratio', tickDelta=12):
	rows         = len(x)
	k_training_all   = estimatePIDModelKalman(x, setpoint)
	xhat_all         = estimateEndingState(k_training_all, x)

	trainingRows = int(rows/2)
	k_training_2ndhalf   = estimatePIDModelKalman(x[trainingRows:], setpoint)
	xhat_2ndhalf         = estimateEndingState(k_training_2ndhalf, x[trainingRows:])


	# initalize the slope with the average slope over the first half.
	[simOut_test_all, timeSim0] = simulatePIDModel(rows, futureRows, k_training_all, x, \
		xhat_all[0], xhat_all[1], xhat_all[2])
	# initalize the slope with the average slope over the first half.
	[simOut_test_2ndhalf, timeSim0] = simulatePIDModel(rows, futureRows, k_training_2ndhalf, x, \
		xhat_2ndhalf[0], xhat_2ndhalf[1], xhat_2ndhalf[2])

	deltaN = N[1]-N[0]
	futureN = N[-1] + np.arange(1, futureRows+1, 1)*deltaN
	#plt.clf()
	plt.plot(N, x + 1.0, 'g-o', linewidth=2, label='Actual')
	plt.plot(futureN, simOut_test_all[:,0] + 1.0, 'b-', linewidth=1, label='Prediction, Trained on All')
	plt.plot(futureN, simOut_test_2ndhalf[:,0] + 1.0, 'k--', linewidth=1, label='Prediction, Trained on 2nd Half')
	plt.plot([min(N)-0.5, max(futureN)+0.5], [1, 1], 'k-', linewidth=1)
	plt.xlim(min(N)-0.5, max(futureN)+0.5)
	plt.xlabel("Year", fontsize=20)
	plt.ylabel(ylabelstr, fontsize=20)
	plt.xticks(range(int(min(N)), int(max(futureN)), tickDelta))
	plt.legend(fontsize=16)
	plt.grid(True)
	plt.show()

	return simOut_test_all, simOut_test_2ndhalf