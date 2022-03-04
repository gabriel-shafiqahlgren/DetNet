#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 10:45:06 2022

@author: gabahl

Visualizing data generated from ggland in python instead of root
"""

import os
import numpy as np
import matplotlib.pyplot as plt


from utils.data_preprocess import load_data
from utils.help_methods import cartesian_to_spherical

#file in data catalogue
NPZ_DATAFILE = os.path.join(os.getcwd(), 'data', 'no_rejection.npz') 


def plot_data_hist(labels, data, close=True, spherical=False):
    """
    Plots E,theta,phi data loaded from npz_datafile in histograms. 
    
    Parameters
    ----------
    data : numpy.ndarray
        Array with det_data (162 crystals).
    labels : numpy.ndarray
        Array with label_data (px py pz).
    close : Boolean, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    fig1 : matplotlib.figure.Figure
        Refrence to energy hist.
    fig2 : matplotlib.figure.Figure
        Refrence to theta hist.
    fig3 : matplotlib.figure.Figure
        Refrence to phi hist.
    fig4 : matplotlib.figure.Figure
        Refrence to sum of all crystal energies hist.    
    fig5 : matplotlib.figure.Figure
        Refrence to number of crystals activated (differs ) hist.
    """
    
    TEXT_SIZE = 20
    NO_BINS= 100
    
    if not spherical:
        labels = cartesian_to_spherical(labels, error=True)
        
    E = labels[::,0::3].flatten()
    theta = labels[::,1::3].flatten()
    phi = labels[::,2::3].flatten()
    
    E_sum_det = data.sum(axis=1)
    XBn_actual = (data != 0).sum(1)
    
    if close:
       plt.close('all')
    
    fig1 = plt.figure(1)
    plt.hist(E, bins=NO_BINS, histtype='step', color='blue', log=True)
    plt.ylabel('N', fontsize=TEXT_SIZE)
    plt.xlabel('$\hat{E}$ [MeV]', fontsize=TEXT_SIZE) 
    
    #removing the angles where no energy was detected theta, phi = 0
    fig2 = plt.figure(2)
    plt.hist(theta[theta!=0], bins=NO_BINS, histtype='step', color='red')
    plt.xticks(np.linspace(0, np.pi, 3),['0','$\pi/2$','$\pi$'])
    plt.ylabel('N', fontsize=TEXT_SIZE)
    plt.xlabel('$\hat{\\theta}$', fontsize=TEXT_SIZE)
    
    fig3 = plt.figure(3)
    plt.hist(phi[phi!=0], bins=NO_BINS, histtype='step', color='green',)
    plt.xticks(np.linspace(0, 2*np.pi, 3),['0','$\pi$','$2\pi$'])
    plt.ylabel('N', fontsize=TEXT_SIZE)
    plt.xlabel('$\hat{\phi}$', fontsize=TEXT_SIZE)
    
    fig4 = plt.figure(4)
    plt.hist(E_sum_det, bins=NO_BINS, histtype='step', color='yellow',)
    plt.ylabel('N', fontsize=TEXT_SIZE)
    plt.xlabel('$\hat{E_{det}}$', fontsize=TEXT_SIZE)
    
    fig5 = plt.figure(5)
    plt.hist(XBn_actual, bins=NO_BINS, histtype='step', color='orange',)
    plt.ylabel('N', fontsize=TEXT_SIZE)
    plt.xlabel('$\hat{XBn}$', fontsize=TEXT_SIZE)
    
    return fig1, fig2, fig3, fig4, fig5
  

# <-- test run here-->
#Add zeros adds completly empty events. data[i]=[0, 0, ... , 0, 0]
#data, labels = load_data(NPZ_DATAFILE, total_portion=1, add_zeros=int(5*1e4))
data, labels = load_data(NPZ_DATAFILE, total_portion=1)  
plot_data_hist(labels, data)






