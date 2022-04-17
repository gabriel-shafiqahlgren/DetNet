#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 21:35:31 2022

@author: gabahl
"""
import os

from tensorflow.keras.optimizers import Adam
from utils.models import FCN, Dense_ResNet

from loss_function.loss import LossFunction

from utils.data_preprocess import load_data

from utils.help_methods import get_momentum_error_dist, cartesian_to_spherical, get_permutation_match
import numpy as np

from utils.plot_methods import plot_predictions_bar_adjustable


def main():
    PORTION_ZEROS = 0.05
    LEARNING_RATE = 1e-4
    NETWORK_TYPES = {'FCN':1, 'Dense_ResNet':2}    
    
    # Change network here
    network_type = NETWORK_TYPES['FCN']

    
    data_file = os.path.join(os.getcwd(), 'data', '3maxmul_0.1_10MeV_500000_clus300.npz')
    data, labels = load_data(data_file, total_portion=1, portion_zeros=PORTION_ZEROS)
    no_inputs = len(data[0])
    no_outputs = len(labels[0])
    
    weights_file = './example/weights.h5'
    
    if network_type == NETWORK_TYPES['FCN']:
        no_layers = 5 
        no_nodes = 124
        model = FCN(no_inputs, no_outputs + 3, no_layers, no_nodes)
    
    
    if network_type == NETWORK_TYPES['Dense_ResNet']:
        no_nodes = 2**7
        no_blocks = 10
        model = Dense_ResNet(no_inputs, no_outputs, no_nodes, no_blocks, no_skipped_layers=3)
    
    max_mult = int(no_outputs / 3)
    loss = LossFunction(max_mult, regression_loss='squared')
    model.compile(optimizer=Adam(lr=LEARNING_RATE), loss=loss.get(), metrics=['accuracy'])
    model.load_weights(weights_file)

    predictions = model.predict(data)
    
    predictions, labels = get_permutation_match(predictions, labels, loss, max_mult)
    
        #Check results
    ME_dist = get_momentum_error_dist(predictions, labels, False)
    print('Total momentum error MME = ', np.sum(ME_dist))
    
    predictions = cartesian_to_spherical(predictions, error=False)
    labels = cartesian_to_spherical(labels, error=False)
    figure, rec_events = plot_predictions_bar_adjustable(predictions, labels, show_detector_angles=True, 
                                                         Ex_max = 10, Ey_max = 14, no_xticks=6, no_yticks=8)

    
if __name__ == "__main__":
    main() 