#! /usr/bin/env python

#
# Script name: numericalSimulation.py
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


matplotlib.rc('xtick', labelsize=14) 
matplotlib.rc('ytick', labelsize=14) 

# main
x = []
init_x = 1.0
init_slope = -0.2
init_sum = 0.0
init_S = [init_x, init_slope, init_sum]
k_est = [-0.05, -0.5, 0.02]
duration = 10
startYear = 0

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


plt.ion()
plt.figure(1)
ax0 = plt.subplot(311)
plt.plot(timeSim, simOut[:,0], 'g-o', linewidth=2, label='Inequity = Ratio-1.0')
plt.xlim(min(timeSim)-0.1, max(timeSim)+0.1)
plt.ylim(-0.2, 1.2)
plt.yticks([0.0, 0.3, 0.6, 0.9, 1.2])
plt.xticks(range(int(min(timeSim)), int(max(timeSim)+1), 1))
ax0.set_xticklabels([])
plt.ylabel('Proportional', fontsize=16)
plt.legend(fontsize=14, loc='lower right')
plt.grid(True)
ax1 = plt.subplot(312)
plt.plot(timeSim, simOut[:,2] , 'b-o', linewidth=2, label='Cumulative Inequity')
plt.xlim(min(timeSim)-0.1, max(timeSim)+0.1)
plt.ylabel('Integral', fontsize=16)
plt.yticks([0.0, 2.0, 4.0, 6.0])
plt.xticks(range(int(min(timeSim)), int(max(timeSim)+1), 1))
ax1.set_yticklabels(["0.0", "2.0", "4.0", "6.0"	])
ax1.set_xticklabels([])
plt.legend(fontsize=14, loc='lower right')
plt.grid(True)
plt.subplot(313)
plt.plot(timeSim, simOut[:,1] , 'r-o', linewidth=2, label='Slope of Inequity')
plt.xlim(min(timeSim)-0.1, max(timeSim)+0.1)
plt.xlabel("Year", fontsize=16)
plt.xticks(range(int(min(timeSim)), int(max(timeSim)+1), 1))
plt.ylabel('Derivative', fontsize=16)
plt.ylim(-0.24, 0.15)
plt.yticks([-0.2, -0.1, 0.0, 0.1])
plt.legend(fontsize=14, loc='lower right')
plt.grid(True)
plt.show()

filename = "sim_kP" + str(k_est[0]) + "_kI" + str(k_est[2]) + "_kD" + str(k_est[01]) + "_initx"+ str(init_x) + "initsum" + str(init_sum) + "initslope" + str(init_slope)

plt.savefig(filename + ".eps")
plt.savefig(filename + ".png")

