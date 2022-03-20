#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  18 13:25:49 2022

@author: gabahl
"""

import os
import numpy as np

from utils.data_preprocess import load_data
from utils.help_methods import cartesian_to_spherical
from utils.help_methods import spherical_to_cartesian
from utils.help_methods import get_permutation_match, get_momentum_error_dist

from loss_function.loss import LossFunction

from utils.plot_methods import plot_predictions



def main():    
    data_file = os.path.join(os.getcwd(), 'data', '3maxmul_0.1_10MeV_500000_clus300.npz')
    data, labels = load_data(data_file, total_portion=1e-1)
    
    predictions, maxmult = addback(data, no_neighbors=10, energy_weighted=True, cluster=False)
    predictions = spherical_to_cartesian(predictions)
    labels = reshapeArrayZeroPadding(labels, labels.shape[0], maxmult*3)
    
    #Finding out the permutation match using prebuild functions
    loss = LossFunction(maxmult, regression_loss='squared')
    predictions, labels = get_permutation_match(predictions, labels, loss, maxmult)
    
    #Check results
    MME = get_momentum_error_dist(predictions, labels, False)
    print('Mean momentum error MME = ', np.mean(MME))
    
    predictions = cartesian_to_spherical(predictions, error=True)
    labels = cartesian_to_spherical(labels, error=True)
    figure, rec_events = plot_predictions(predictions, labels, show_detector_angles=True)


class Crystal:
    def __init__(self, index, crystal_type, E, theta, phi, visited, neighbors):
        self.index = index
        self.E = E
        self.theta = theta * (np.pi/180) #convestions to theta [0, pi], phi [0, 2pi]
        if phi < 0:
            self.phi = (360 + phi) * (np.pi/180)
        else:
            self.phi = phi *  (np.pi/180)
        self.visited = visited 
        self.neighbors = neighbors
        self.crystal_type = crystal_type
        
    def __repr__(self):
        return f'index:{self.index}, crystal_type:{self.crystal_type}, E:{self.E}, theta:{self.theta}, phi:{self.phi}, visited:{self.visited}, neighbors:{self.neighbors}'

    def setEnergy(self, E):
        self.E = E

    def visit(self):
        self.visited = True
        return True
    
    def unvisit(self):
        self.visited = False
        return False    
   
    
def createCrystalBall():
    crystalBall = {}
    with open('data/geom_xb.txt', 'r') as infile:
        for line in infile:
            line = line.lstrip('XB _ GEOM ( ')
            line = line.split(')')[0] #removes everthing after ')'
            attributes = line.split(',')
            crystal = Crystal(
                int(attributes[0]), #index
                attributes[1], #crystal type
                0, #E = 0
                float(attributes[2]), #theta
                float(attributes[3]), #phi
                False,
                [int(n) for n in attributes[5:] if int(n)!=0]  #Assuming 0 in geom_xb.txt means no neighbor
                )
            crystalBall[crystal.index] = crystal
    return crystalBall


def getDepositsSortedList(event):
    depositsDict = {}
    for energyDeposit in enumerate(event, 1):
        if energyDeposit[1] > 0:
            depositsDict[energyDeposit[0]] = energyDeposit[1]    
    return sorted(depositsDict.items(), key=lambda x:x[1],reverse=True)


def setCrystalBall(crystalBall, event):
    for energyDeposit in enumerate(event, 1):
        crystalBall[energyDeposit[0]].setEnergy(energyDeposit[1])
        crystalBall[energyDeposit[0]].unvisit()


def findHit(crystalBall, ci, no_neighbors, weighted, cluster): #ci crystal index
    E, theta, phi = 0, 0, 0
    E, theta, phi = sumNeighbors(crystalBall, ci, no_neighbors, weighted, cluster, E, theta, phi)
    if weighted:
        theta = theta/E
        phi = phi/E
    else:
        theta = crystalBall[ci].theta
        phi = crystalBall[ci].phi
    return E, theta, phi


def sumNeighbors(crystalBall, ci, nn, weighted, cluster, E, theta, phi):
    crystal = crystalBall[ci]
    if not crystal.visited:
        crystal.visit()
        E += crystal.E
        if weighted:
            theta += crystal.E*crystal.theta
            phi += crystal.E*crystal.phi
        if not cluster:
            if nn > 0:
                for neighbor_ci in crystal.neighbors:
                    E, theta, phi = sumNeighbors(crystalBall, neighbor_ci, nn-1, weighted, cluster, E, theta, phi)
        if cluster:
            if nn > 0 and crystal.E != 0.:
                for neighbor_ci in crystal.neighbors:
                    E, theta, phi = sumNeighbors(crystalBall, neighbor_ci, nn-1, weighted, cluster, E, theta, phi)
    return E, theta, phi


def listOfListsToPaddedZeroArray(lst, maxmult):
    no_events = len(lst)
    predictions = np.zeros([no_events, maxmult*3])
    for i in range(no_events):
        j = -1
        for val in reversed(lst[i]):
            predictions[i, j] = val
            j -= 1
    return predictions


def reshapeArrayZeroPadding(array, i, j):
    A = np.zeros([i, j])
    for k in range (i):
        l = -1
        for val in reversed(array[k]):
            A[k, l] = val
            l -= 1
    return A


def addback(data, no_neighbors=1, energy_weighted=False, cluster=False):
    """
    Parameters
    ----------
    data : np.array
        Array of shape (i, 162) of event crystal values
    no_neighbors : int, optional
        Max number of neightbors if max energy crystal hit included in the 
        algorithm. The default is 1.
    energy_weighted : boolean, optional
        Do a weighted sum of energy and theta/phi instead of taking the angles
        of the max energy crystal as first hit. The default is False.
    cluster : boolean, optional
        Only visit neighbors of activated crystals. The default is False.

    Returns
    -------
    predictions : np.array
        Predicted E, theta, phi values for each event (row).
    maxmult : int
        Max number of particles predicted by addback.
    """
    
    
    crystalBall = createCrystalBall()
    maxmult = 0
    predictions=[]
    for event in data:
        setCrystalBall(crystalBall, event)
        deposits = getDepositsSortedList(event)
        predictionRow = []
        event_maxmult = 0
        for deposit in deposits:
            if not crystalBall[deposit[0]].visited: 
                E, theta, phi = findHit(crystalBall, deposit[0], no_neighbors, energy_weighted, cluster)
                predictionRow.append(E)
                predictionRow.append(theta)
                predictionRow.append(phi)
                event_maxmult +=1
        maxmult = max([maxmult, event_maxmult])
        predictions.append(predictionRow)
    print(f'max_mult = {maxmult}')
    predictions = listOfListsToPaddedZeroArray(predictions, maxmult)
    print('Addback complete')
    return predictions, maxmult


if __name__ == "__main__":
    main()



    
