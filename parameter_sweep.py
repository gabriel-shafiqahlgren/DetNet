#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 13:03:19 2022

Basically a way to iterate between the different parameters.

@author: gabahl
"""

import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from utils.models import FCN

from loss_function.loss import LossFunction

from utils.plot_methods import plot_predictions
from utils.plot_methods import plot_loss

from utils.data_preprocess import load_data
from utils.data_preprocess import get_eval_data

from utils.help_methods import save
from utils.help_methods import get_permutation_match
from utils.help_methods import cartesian_to_spherical
from utils.help_methods import get_measurement_of_performance
from utils.help_methods import get_momentum_error_dist

import numpy as np


## --------------- Constants
TOTAL_PORTION = 1      #portion of file data to be used, (0,1]
EVAL_PORTION = 0.2      #portion of total data for final evalutation (0,1)
VALIDATION_SPLIT = 0.1  #portion of training data for epoch validation

## --------------- Sweep parameters
ls_no_layers = [*range(10)] + [*range(10,31,5)]
ls_no_nodes = [64]
ls_no_epochs = [250]
ls_no_batch_size = [2**8]
ls_learning_rate = [1e-4]


## ----------- Load and prepare data for training/evaluation -------------------
onlyfiles = [f for f in os.listdir('data') if os.path.isfile(os.path.join('data', f))]
for filename in onlyfiles:
    print(filename)

filename = input('Which data file ("def" -> "XB_mixed_data.npz")?: ')
if filename=='def':
    filename='XB_mixed_data.npz'
    print('You entered default')
else: 
    print("You entered: " + filename)


save_filename = input('Save sweep as ("def" -> <filename>.txt?: ')
if save_filename == 'def':
    save_filename = filename.replace('npz','txt')
print('Data will be saved in ' + save_filename)

no_iter = int(input('How many times do you want the script to run?: '))
if no_iter < 0 or no_iter > 10:
    no_iter = 1
    print('Input is not a whole number greater than zero and less than 10.'+'\n'+'The script will run one time.')
    

f = open(save_filename, 'a')
f.write(filename+'\n'+'MME, no_layers, no_nodes, no_epochs, batch_size, lr_rate'+'\n')
f.close()


npz_datafile = os.path.join(os.getcwd(), 'data', filename)
#load simulation data. OBS. labels need to be ordered in decreasing energy!
data, labels = load_data(npz_datafile, TOTAL_PORTION)

#detach subset for final evaluation, train and eval are inputs, train_ and eval_ are labels
train, train_, eval, eval_ = get_eval_data(data, labels, eval_portion=EVAL_PORTION)

for i in range(no_iter):
    f = open(save_filename, 'a')
    f.write('Iteration ' + f'{i +1}' + '\n')
    f.close()
    def build_FCN_fix_param(train, train_, NO_LAYERS, NO_NODES, \
                        NO_EPOCHS, BATCH_SIZE, LEARNING_RATE, VALIDATION_SPLIT):
    ## ---------------------- Build the neural network -----------------------------
    
        # Derived parameters
        no_inputs = len(train[0])
        no_outputs = len(train_[0])
        model = FCN(no_inputs, no_outputs, NO_LAYERS, NO_NODES)
    
        # select mean squared error as loss function
        max_mult = int(no_outputs / 3)
        loss = LossFunction(max_mult, regression_loss='squared')
    
        #compile the network
        model.compile(optimizer=Adam(lr=LEARNING_RATE), loss=loss.get(), metrics=['accuracy'])

        ## ----------------- Train the neural network and plot results -----------------
        #training = model.fit(
        model.fit(train, train_,
                         epochs=NO_EPOCHS,
                         batch_size=BATCH_SIZE,
                         validation_split=VALIDATION_SPLIT,
                         callbacks=[EarlyStopping(monitor='val_loss', patience=3)])
        return model, loss, max_mult



    for no_layers in ls_no_layers:
        for no_nodes in ls_no_nodes:
            for no_epochs in ls_no_epochs:
                for batch_size in ls_no_batch_size:
                    for lr_rate in ls_learning_rate:
                        model, loss, max_mult = build_FCN_fix_param(train, train_, no_layers, no_nodes, no_epochs,\
                                        batch_size, lr_rate, VALIDATION_SPLIT)
                    
                        # get predictions on evaluation data
                        predictions = model.predict(eval)
                       
                        # return the combination that minimized the loss function (out of max_mult! possible combinations)
                        predictions, eval_ = get_permutation_match(predictions, eval_, loss, max_mult)
                     
                        # print the error in E, theta and phi a
                        get_measurement_of_performance(predictions, eval_, False)
                       
                        #calculate the mean momentum error (MME)
                        MME = get_momentum_error_dist(predictions, eval_, False)
                        MME = np.mean(MME)
                    
                        f = open(save_filename, 'a')
                        f.write(f'{MME}, {no_layers}, {no_nodes}, {no_epochs}, {batch_size}, {lr_rate} \n')
                        f.close()
                    
                    
                        
    
    
