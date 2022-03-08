#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 13:11:27 2022

Script that loads the file saved with parameter sweep.

@author: gabahl
"""

import matplotlib.pyplot as plt
import pandas as pd

TEXT_SIZE = 15

font = {'family' : 'STIXGeneral',
        'weight' : 'normal',
        'size'   : TEXT_SIZE}
plt.rc('font', **font)
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

df = pd.read_csv("./DetNet/ann_iterations/parasweepnm/data_matrix.csv")
df_to_doct = df.to_dict()

x = 'Patience' # Key from cvs file to have as x axis
y = 'Mean of mean momentum error' # Use as key to 
yerr = 'Std of mean momentum error'

ls_x = df_to_doct[x].values()
ls_y = df_to_doct[y].values()
ls_yerr = df_to_doct[yerr].values()

label = 'w128 d5' # Legend label; network specifics 
plt.rcParams["figure.figsize"] = (14,8)
plt.errorbar(ls_x, ls_y, yerr=ls_yerr, label=label)

plt.grid() # Optional
plt.legend()
plt.xlabel(x)
plt.ylabel(y + ' (N =  3)')
plt.savefig(x+'.png')