#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28
Runs the addback routine and FCN on several evaluation data files. Stores the MME 
values in a .txt-file. This is only working for 

Comment> Would be alot easier if the models were saved rather than their weights.
@author: gabahl, bjoj
"""
import os
import numpy as np
import glob as glob

from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from utils.models import FCN, Dense_ResNet

from pathlib import Path

from loss_function.loss import LossFunction

from utils.data_preprocess import load_data

from utils.help_methods import get_momentum_error_dist, cartesian_to_spherical, get_permutation_match
from utils.help_methods import spherical_to_cartesian, get_permutation_match_with_permutations

from addback import addback
from addback import reshapeArrayZeroPadding

def main():
    PORTION_ZEROS = 0.05
    LEARNING_RATE = 1e-4
    NETWORK_TYPES = {'FCN':1, 'Dense_ResNet':2}    
    
    # Change network here, probably need to update the code to function properly for anything but FCN
    network_type = NETWORK_TYPES['FCN']

    # The directory where all weights are stored for FCNs
    model_directory = '/cephyr/NOBACKUP/groups/snic2022-5-74/FCN_sweeps/DetNet/saved_models/'

    # The directory where evaluation files are stored
    data_directory = 'evaluation_files'
    data_files = Path(data_directory).rglob('*.npz')
    data_dir = glob.glob(os.path.join(data_directory, '*.npz'))

    # The file that the MME:s will be stored
    save_file_name = 'mme_table.txt'

    # Write what evaluation files are used as header
    save_file = open(save_file_name, 'a')
    save_file.write('Network type ')
    for data_file in sorted(data_files):
        save_file.write(str(data_file) + ' ')
        save_file.write('\n')
    save_file.close()

    # Go through all evaluation files with the addback routine
    save_file = open(save_file_name, 'a')
    save_file.write('Addback ')
    save_file.close()
    for data_file in sorted(data_dir):
        print('test')
        data, labels = load_data(data_file, total_portion=1, portion_zeros=PORTION_ZEROS)

        predictions, maxmult = addback(data, no_neighbors=1, energy_weighted=True, cluster=False)
        predictions = spherical_to_cartesian(predictions)

        # Minimize the MME/event
        predictions, labels = get_permutation_match_with_permutations(predictions, labels)
        labels = reshapeArrayZeroPadding(labels, labels.shape[0], maxmult*3)
                
        ME_dist = get_mse(predictions, labels, False)
        MME = np.mean(ME_dist)
        save_file = open(save_file_name, 'a')
        save_file.write(str(MME) + ' ')
        print('MME ' + str(MME))
        save_file.close()
        eval_name1 = data_file
        eval_name1 = eval_name1.replace('.npz','')
        eval_name1 = eval_name1.replace('/','')
        dist_addback = open('addback_' + eval_name1 + '_mme_dist.txt', 'a')
        dist_addback.write(str(ME_dist))
        dist_addback.close()       
    
    # Sweep through the FCNs
    #for model_file in os.scandir(model_directory):
    # if model_file.path.endswith(model_ext):
    for model_name in sorted(os.listdir(model_directory)):
        if os.path.isdir(model_name)==False:
            model = keras.models.load_model(model_directory + model_name, compile=False)
            save_file = open(save_file_name, 'a')
            save_file.write('\n' + str(model_name) + ' ')
            save_file.close()
        
            # Go through all evaluation files for each FCN model
            for data_file in sorted(data_dir):
                data, labels = load_data(data_file, total_portion=1, portion_zeros=PORTION_ZEROS)
                no_outputs = len(labels[0])
                max_mult = int(no_outputs / 3)

                # loss = LossFunction(max_mult, regression_loss='squared') 
                loss = LossFunction(max_mult, regression_loss='absolute') 
        
                # Make reconstructions
                predictions = model.predict(data)
                predictions, labels = get_permutation_match(predictions, labels, loss, max_mult)
    
                #Check results
                ME_dist = get_mse(predictions, labels, False)
                MME = np.mean(ME_dist)
                save_file = open(save_file_name, 'a')
                save_file.write(str(MME)  + ' ')
                print('\nMME ' + str(MME))
                save_file.close()
                eval_name = data_file
                eval_name = eval_name.replace('.npz','')
                eval_name = eval_name.replace('/','')
                dist_ANN = open(model._name + '_' + str(model_name) + '_' + eval_name + '_mme_dist.txt', 'a')
                dist_ANN.write(str(ME_dist))
                dist_ANN.close()
                print('Total momentum error = ', np.sum(ME_dist))
                print('MME = ', str(MME))
    print('\nDone')

# Returns an array of mse/event
def get_mse(prediction, label, spherical=True):
    if spherical:
        prediction = spherical_to_cartesian(prediction)
        label = spherical_to_cartesian(label)

    prediction_list = lambda q: prediction[::,q::3]
    label_list = lambda q: label[::,q::3]

    momentum_tensor = np.array([prediction_list(0) - label_list(0), prediction_list(1) - label_list(1), prediction_list(2)- label_list(2)])

    mult = int(prediction.shape[1]/3)
    mse = [np.sum([np.linalg.norm(momentum_tensor[:,i,j]) for j in range(mult)]) for i in range(momentum_tensor.shape[1])]
    return mse

if __name__ == "__main__":
    main() 
