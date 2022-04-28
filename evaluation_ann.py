#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28
Runs the addback routine and FCN on several evaluation data files. Stores the MME results in a file.

@author: gabahl, bjoj
"""
import os

from tensorflow.keras.optimizers import Adam
from utils.models import FCN, Dense_ResNet

from pathlib import Path

from loss_function.loss import LossFunction

from utils.data_preprocess import load_data

from utils.help_methods import get_momentum_error_dist, cartesian_to_spherical, get_permutation_match
import numpy as np

from utils.plot_methods import plot_predictions_bar

def main():
    PORTION_ZEROS = 0.05
    LEARNING_RATE = 1e-4
    NETWORK_TYPES = {'FCN':1, 'Dense_ResNet':2}    
    
    # Change network here, probably need to update the code to function properly for anything but FCN
    network_type = NETWORK_TYPES['FCN']

    # The directory where all weights are stored for FCNs
    evaluation_directory = 'network_parameters'
    weight_files = Path(evaluation_directory).glob('*.h5')

    # The directory where evaluation files are stored
    data_directory = 'evaluation_files'
    data_files = Path(data_directory).glob('*.npz')

    # The file that the MME:s will be stored
    save_file_name = 'mme_table.txt'

    # Write what evaluation files are used as header
    save_file = open(save_file_name, 'a')
    save_file.write('Network type ')
    for data_file in data_files
        save_file.write(data_file + ' ')
    save_file.write('\n')
    save_file.close()
    
    # Sweep through the FCNs
    for weight_file in weight_files
        # Go through all evaluation files for each FCN model
        for data_file in data_files 
            data, labels = load_data(data_file, total_portion=1, portion_zeros=PORTION_ZEROS)
            no_inputs = len(data[0])
            no_outputs = len(labels[0])
        
            # Optimized FCN
            if network_type == NETWORK_TYPES['FCN']:
                no_layers = 6 
                no_nodes = 1200
                model = FCN(no_inputs, no_outputs, no_layers, no_nodes)
        
            # This needs to be updated
            if network_type == NETWORK_TYPES['Dense_ResNet']:
                no_nodes = 2**7
                no_blocks = 10
                model = Dense_ResNet(no_inputs, no_outputs, no_nodes, no_blocks, no_skipped_layers=3)
        
            max_mult = int(no_outputs / 3)
            loss = LossFunction(max_mult, regression_loss='squared')
            model.compile(optimizer=Adam(lr=LEARNING_RATE), loss=loss.get(), metrics=['accuracy'])
        
            model.load_weights(weight_files)

            # Make reconstructions
            predictions = model.predict(data)
    
            predictions, labels = get_permutation_match(predictions, labels, loss, max_mult)
    
            #Check results
            ME_dist = get_momentum_error_dist(predictions, labels, False)
            print('Total momentum error MME = ', np.sum(ME_dist))
            print('MME = ', np.sum(ME_dist)/len(predictions)/3)

            # These are not needed for this script
            #predictions = cartesian_to_spherical(predictions, error=False)
            #labels = cartesian_to_spherical(labels, error=False)

    
if __name__ == "__main__":
    main() _angles=True)

    
if __name__ == "__main__":
    main() 
