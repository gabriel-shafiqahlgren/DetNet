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

from model_evaluation import evaluate_model

def main():
    PORTION_ZEROS = 0.05
    LEARNING_RATE = 1e-4
    NETWORK_TYPES = {'FCN':1, 'Dense_ResNet':2}    
    
    # Change network here
    network_type = NETWORK_TYPES['FCN']
    no_inputs = 162
    no_outputs = 9
    
    weights_file = './ann_iterations/FCN_layer_6_bs_10/data_point26/2022-04-09-04;33;35/weights.h5'
    
    if network_type == NETWORK_TYPES['FCN']:
        no_layers = 6 
        no_nodes = 1200
        model = FCN(no_inputs, no_outputs, no_layers, no_nodes)
    
    
    if network_type == NETWORK_TYPES['Dense_ResNet']:
        no_nodes = 2**7
        no_blocks = 10
        model = Dense_ResNet(no_inputs, no_outputs, no_nodes, no_blocks, no_skipped_layers=3)
    
    max_mult = int(no_outputs / 3)
    loss = LossFunction(max_mult, regression_loss='squared')
    model.compile(optimizer=Adam(lr=LEARNING_RATE), loss=loss.get(), metrics=['accuracy'])
    model.load_weights(weights_file)

    #Will perform a lot of evaluations and save to folder
    evaluate_model(model)

    
if __name__ == "__main__":
    main() 
