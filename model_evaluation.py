#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 14:52:41 2022

@author: gabahl
"""

import os
from utils.data_preprocess import load_data
from utils.plot_methods import plot_predictions_bar_adjustable
from utils.help_methods import get_permutation_match, get_measurement_of_performance, cartesian_to_spherical, get_permutation_match_with_permutations
from loss_function.loss import LossFunction
from addback import reshapeArrayZeroPadding
import numpy as np

# Data paths
EVAL_FOLDER = '/cephyr/NOBACKUP/groups/snic2022-5-74/eval'
FIGURE_FOLDER = os.path.join(os.getcwd(), 'eval_figures')
LARGE_EVAL = 'eval10_train.npz' # Need large eval file 
NOISE_EVAL = 'eval10_train.npz'
LOW_ENERGY = 'eval5_all.npz' # Need low energy without all events
HIGH_ENERGY = 'eval_15.npz'
#MULT_1 = 
MULT_2 = 'eval10_2gamma_all.npz' 
MULT_4 = 'eval10_4gamma_all.npz'
MULT_5 = 'eval10_5gamma_all.npz'
REGRESSION_LOSS = 'absolute'



def evaluate_model(model):
    evaluate_new_eval_data(model)
    evaluate_with_noise(model)
    evaluate_lower_energies(model)
    evaluate_higher_energies(model)
    evaluate_lower_multiplicities(model)
    evaluate_higher_multiplicities(model)
    return


def predict_model(model, data, labels, regression_loss):
    
    predictions = model.predict(data)
    
    max_mult = int(labels.shape[1]/3)
    loss = LossFunction(max_mult, regression_loss=regression_loss)
   
    predictions, labels = get_permutation_match(predictions, labels, loss, max_mult)
    
    return predictions, labels


def plot_and_print_performance(predictions, labels, message, Ex=10, Ey=10):
    mop = get_measurement_of_performance(predictions, labels, spherical=False)
    print(message + '\n' + str(mop))
    
    #predictions = cartesian_to_spherical(predictions, error=True)
    #labels = cartesian_to_spherical(labels, error=True)
    
    fig, events =  plot_predictions_bar_adjustable(predictions, labels, show_detector_angles=True,
                                                         Ex_max=Ex, Ey_max=Ey, epsilon=0.1, remove_zero_angles=True)
    fig.savefig(os.path.join(FIGURE_FOLDER, message + '.png'))
    return fig, events
    
def evaluate_new_eval_data(model):
    data, labels = load_data(os.path.join(EVAL_FOLDER, LARGE_EVAL), total_portion=1)
    
    predictions, labels = predict_model(model, data, labels, regression_loss=REGRESSION_LOSS)
    
    return plot_and_print_performance(predictions, labels, 'New_eval_data')

#TODO
def evaluate_with_noise(model):
    data, labels = load_data(os.path.join(EVAL_FOLDER, NOISE_EVAL), total_portion=1)

    for i in range(data.shape[0]):
        data[i] + np.random.normal(0.150,0.100, data.shape[1]) # mu=150 keV sigma=100keV
    predictions, labels = predict_model(model, data, labels, regression_loss=REGRESSION_LOSS)
    
    return plot_and_print_performance(predictions, labels, 'Data_with_noise')
    

def evaluate_lower_energies(model):
    data, labels = load_data(os.path.join(EVAL_FOLDER, LOW_ENERGY), total_portion=1)

    predictions, labels = predict_model(model, data, labels, regression_loss=REGRESSION_LOSS)
    
    return plot_and_print_performance(predictions, labels, 'Lower_energies', Ex=5)
    

def evaluate_higher_energies(model):
    data, labels = load_data(os.path.join(EVAL_FOLDER, HIGH_ENERGY), total_portion=1)
    
    predictions, labels = predict_model(model, data, labels, regression_loss=REGRESSION_LOSS)
    
    return plot_and_print_performance(predictions, labels, 'Higher_energies', Ex=15, Ey=15)

### Not finished

def evaluate_lower_multiplicities(model):
    data, labels = load_data(os.path.join(EVAL_FOLDER, MULT_2), total_portion=1)
    
    predictions = model.predict(data)
    maxmult = int(predictions.shape[1]/3)
    predictions, labels = get_permutation_match_with_permutations(predictions, labels)
    labels = reshapeArrayZeroPadding(labels, labels.shape[0], maxmult*3)
    
    return plot_and_print_performance(predictions, labels, 'm_2')

def evaluate_higher_multiplicities(model):
    data, labels = load_data(os.path.join(EVAL_FOLDER, MULT_4), total_portion=1)
    
    predictions = model.predict(data)
    maxmult = int(labels.shape[1]/3)
    predictions = reshapeArrayZeroPadding(predictions, predictions.shape[0], maxmult*3)
    predictions, labels = get_permutation_match_with_permutations(predictions, labels)

    return plot_and_print_performance(predictions, labels, 'm_4')
    




