#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 13:11:27 2022

Script that loads the file saved with parameter sweep.

@author: gabahl
"""


import numpy as np
import matplotlib.pyplot as plt



#get the ttl_port figures.
#Check firsts row of file for exact parameter values.
arr = np.loadtxt('ann_iterations/lr_rate_sweep2/data_matrix.csv', delimiter=',', skiprows=1)
plt.rcParams["figure.figsize"] = (14,8)



lr_rate = np.log10(arr[:,9]) 
mean_MME4_128 = arr[:, 3]
std_MME4_128 = arr[:, 4]

lw = 2
plt.errorbar(lr_rate, mean_MME4_128, yerr=std_MME4_128, label='w128 d4')


plt.legend()
plt.xlabel("$log_{10}(learning rate)$")
plt.ylabel("mean MME (N=3)")

plt.show()



