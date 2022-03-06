#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 13:03:19 2022

Script to iterate between the different parameters. 

Requires two data files:
    Variable data for training which is split evenly across the number of iterations
    Constant for testing which is the same across all iterations
    
!!!OBS!!!
Make sure that the test file and load file have the same maxmimum multiplicity
"""

import os
import sys

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from utils.models import FCN

from loss_function.loss import LossFunction

from utils.data_preprocess import load_data
from utils.help_methods import check_folder

from utils.help_methods import get_permutation_match
from utils.help_methods import get_measurement_of_performance

from time import gmtime, strftime, time
from datetime import timedelta, datetime
from utils.help_methods import get_mem_use
from utils.help_methods import get_gpu_memory
from utils.help_methods import save_dictionary_csv
from utils.help_methods import save_figs
from numpy import mean, std


## --------------- Constants
VALIDATION_SPLIT = 0.1  #portion of training data for epoch validation
SWEEP_DATA_FOLDER = 'ann_iterations'
NUMERIC_DATAFILE_NAME = 'numeric_data.csv'
STRING_DATAFILE_NAME = 'info.csv'
LOAD_FILENAME_EVAL = '3maxmul_0.1_10MeV_500000_clus300.npz' #Constant file for testing
PORTION_ZEROS = 0.05 #portion of loaded data to be made into zero events
COLLECTIVE_NUMERIC_DATA_FILENAME = 'data_matrix.csv'
LOAD_FILE_NAME = '3maxmul_0.1_10MeV_3000000_clus300.npz'
ITERATIONS_PER_DATA_POINT = 3
SWEEP_NAME = sys.argv[1]

## --------------- Sweep parameters
ls_total_port = [1] #portion of file data to be used, (0,1]
ls_no_layers = [0]
ls_no_nodes = [0]
ls_no_epochs = [1]
ls_no_batch_size = [2**7]
ls_learning_rate = [1e-4]
ls_patience = [4] #def was 3

## Setting functions 

def input_data_filename():
    ## Prints all files 
    all_files = [f for f in os.listdir('data') if os.path.isfile(os.path.join('data', f))]
    for filename in all_files:
        print(filename)
    
    load_filename = input('Load data from ("def" -> "XB_mixed_data.npz")?: ')
    if load_filename=='def':
        load_filename='XB_mixed_data.npz'
        print('You entered default')
    else: 
        print("You entered: " + load_filename)
    return load_filename


def input_sweep_folder_name(load_filename, sweep_name):
    if sweep_name == 'def':
        sweep_name = load_filename.replace('.npz','')
    mpath = SWEEP_DATA_FOLDER + '/' + sweep_name
    print('Sweeps will be saved in ' + mpath)
    if os.path.isdir(mpath):
        print('Directory already exists.')
    else:
        os.mkdir(mpath)
    return mpath + '/'

def input_number_of_iterations():
    no_iter = int(input('Samples of each data point: '))
    if no_iter < 0 or no_iter > 10:
        no_iter = 1
        print('Input is not a whole number greater than zero or less than 10.'+'\n'+'The script will run one time.')
    return no_iter


def get_metrics(dMetrics, batch_size, no_nodes, no_layers, lr_rate, no_epochs, load_filename, sweep_folder_path, data):
        
    minMax = lambda x: [round(min(x), 3), round(mean(x), 3), round(max(x), 3)]
    mean_loss = round(mean(dMetrics['Loss']), 4)
    std_loss = round(std(dMetrics['Loss']), 4)
    mean_mme = round(mean(dMetrics['P Mean']), 4)  #mme is mean momentum error for the trained model. Mean is with respect to different model weights
    std_mme = round(std(dMetrics['P Mean']), 4)
    mean_std_mme = round(mean(dMetrics['P Std']), 4)
    mean_epochs = round(mean(dMetrics['Epochs']), 4)
    max_epochs = max(dMetrics['Epochs'])
    avg_training_time_min = round(mean(dMetrics['Training time'])/60 , 4)
    
    number_of_events = len(data[:, 0])
    
    ls_MMM_memory_use = minMax(dMetrics['Memory use'])
    ls_MMM_VRAM_use = minMax(dMetrics['GPU memory use'])
    
    dict_numeric_data_point_mean = {'Iterations': dMetrics['Iterations'], 
                                    'Mean loss': mean_loss, 
                                    'Std loss': std_loss,
                                    'Mean of mean momentum error': mean_mme,
                                    'Std of mean momentum error': std_mme,
                                    'Mean of std momentum error': mean_std_mme, 
                                    'Batch size': batch_size, 
                                    'Nodes': no_nodes, 
                                    'Layers': no_layers,
                                    'Learning rate': '{:.2e}'.format(lr_rate), 
                                    'Epochs mean': mean_epochs, 'Epochs stop':no_epochs, 'Epochs max':max_epochs, 
                                    'Patience': patience, 
                                    'Average training time': avg_training_time_min,
                                    'Number of events': number_of_events, 
                                    'Memory GB min': ls_MMM_memory_use[0],
                                    'Memory GB mean': ls_MMM_memory_use[1],
                                    'Memory GB max': ls_MMM_memory_use[2],
                                    'VRAM GB min': ls_MMM_VRAM_use[0],
                                    'VRAM GB mean': ls_MMM_VRAM_use[1],
                                    'VRAM GB max': ls_MMM_VRAM_use[2]
                                    }
    
    # Probably wont work if the optimizer is changed from Adam. 
    # Since it needs the .decay, .beta_1, .beta_2 attributes.    
    dict_str_info_data_point_mean = {'Network': model._name, 
                                     'Optimizer': [optimizer._name, 
                                                   float(optimizer.decay), 
                                                   float(optimizer.beta_1),
                                                   float(optimizer.beta_2)], 
                                     'Regression loss': regression_loss,
                                     'Data file': load_filename, }
    return dict_numeric_data_point_mean, dict_str_info_data_point_mean



#load_filename = input_data_filename()
load_filename = LOAD_FILE_NAME

sweep_folder_path = input_sweep_folder_name(load_filename, SWEEP_NAME)

#iterations = input_number_of_iterations()
iterations = ITERATIONS_PER_DATA_POINT

# Hope ypu have enough RAM
npz_datafile = os.path.join(os.getcwd(), 'data', load_filename)
npz_datafile_eval = os.path.join(os.getcwd(), 'data', LOAD_FILENAME_EVAL)

for total_port in ls_total_port:
    #load simulation data for training. OBS. labels need to be ordered in decreasing energy!
    train_full, train_full_ = load_data(npz_datafile, total_port, portion_zeros=PORTION_ZEROS)
    
    #
    len_data = len(train_full_)
    n = int(len_data/iterations)
    
    #loat simulation data for evaluation. -/-
    eval, eval_ = load_data(npz_datafile_eval, total_portion=1, portion_zeros=PORTION_ZEROS)

    for no_layers in ls_no_layers:
        for no_nodes in ls_no_nodes:
            for no_epochs in ls_no_epochs:
                for batch_size in ls_no_batch_size:
                    for lr_rate in ls_learning_rate:
                        for patience in ls_patience:
                            # Dict to save values from all iterations of the data point
                            dMetrics = {'Loss': [], 
                                        'P Std': [], 
                                        'P Mean': [], 
                                        'Training time': [], 
                                        'Epochs': [],
                                        'Memory use': [], 
                                        'GPU memory use': [], 
                                        'Iterations': 0
                                        }
                            
                            #Make directory for this particular data point
                            data_point_path = check_folder(sweep_folder_path + 'data_point')
                            os.mkdir(data_point_path)
                            
                            for i in range(iterations):
                                print('------------ Iteration: ' + str(i+1) + '/' + str(iterations) + ' ------------')
                                
                                #split the traning data for each iteration. 
                                train = train_full[i*n:(i+1)*n,:] 
                                train_ = train_full_[i*n:(i+1)*n,:]
                                
                    
                                ## ---------------------- Build the neural network -----------------------------
    
                                # Derived parameters
                                no_inputs = len(train[0])
                                no_outputs = len(train_[0])
                                model = FCN(no_inputs, no_outputs, no_layers, no_nodes)
                                
                                regression_loss = 'squared'
                                max_mult = int(no_outputs / 3)
                                # select mean squared error as loss function
                                loss = LossFunction(max_mult, regression_loss=regression_loss)
                                
                                optimizer=Adam(lr=lr_rate)
                                
                                #compile the network
                                model.compile(optimizer, loss=loss.get(), metrics=['accuracy'])
                            
                                ## ----------------- Train the neural network and plot results -----------------
                                start_time = time()
                                training = model.fit(train, train_,
                                                     epochs=no_epochs,
                                                     batch_size=batch_size,
                                                     validation_split=VALIDATION_SPLIT,
                                                     callbacks=[EarlyStopping(monitor='val_loss', patience=patience)])
                                training_time = time() - start_time
                                
                                dMetrics['Training time'].append(training_time)
                                print("Training time> --- %s ---" % timedelta(seconds=round(dMetrics['Training time'][i])))
                                
                                # get predictions on evaluation data
                                predictions = model.predict(eval)
    
                                # return the combination that minimized the loss function (out of max_mult! possible combinations)
                                predictions, eval_ = get_permutation_match(predictions, eval_, loss, max_mult)
    
                                # print the error in E, theta and phi a
                                meas_perf = get_measurement_of_performance(predictions, eval_, spherical=False)
                                
                                dMetrics['P Mean'].append(meas_perf['momentum mean'])
                                dMetrics['P Std'].append(meas_perf['momentum std'])
                                dMetrics['Loss'].append(training.history['loss'][-1]) # Add the final loss
                                dMetrics['Memory use'].append(round(get_mem_use(), 4))
                                dMetrics['GPU memory use'].append(round(get_gpu_memory(), 4))
                                dMetrics['Epochs'].append(training.epoch[-1] + 1)           
                                dMetrics['Iterations'] +=1
                                
                                #
                                current_time = strftime("%Y-%m-%d %H:%M:%S", gmtime()).replace(" ", "-").replace(':',';')
                                iteration_path = data_point_path + current_time
                                path_to_folder = save_figs(iteration_path, figs=[], model=model)
                                
                                #data saved collectively for each iteration
                                dct_Data = {'Loss': training.history['loss'][-1], 
                                            'P mean': meas_perf['momentum mean'], 
                                            'P std': meas_perf['momentum std'] ,
                                            'Batch size': batch_size, 
                                            'Nodes': no_nodes, 
                                            'Layers': no_layers, 
                                            'Learning rate': '{:.2e}'.format(lr_rate),
                                            'Epochs': [training.epoch[-1] + 1, no_epochs], 
                                            'Network': model._name, 
                                            'Optimizer': [optimizer._name, float(optimizer.decay), float(optimizer.beta_1),float(optimizer.beta_2)],
                                            'Regression loss': regression_loss, 
                                            'Training time': timedelta(seconds=round(training_time)), 
                                            'Data file': load_filename,
                                            'Number of events': len(train_full[:,0]), 
                                            'Memory GB': round(get_mem_use(), 4),
                                            'VRAM GB': round(get_gpu_memory(), 4) 
                                            }
                                
                                save_dictionary_csv(os.path.join(path_to_folder, 'data.csv'), dct_Data)
                            
                                
                            dict_numeric_data_point_mean, dict_str_info_data_point_mean = \
                                get_metrics(dMetrics, batch_size, no_nodes, no_layers, lr_rate, no_epochs, load_filename, \
                                            sweep_folder_path, train_full)
                            
                            #Saving data to data point filder
                            save_dictionary_csv(data_point_path + NUMERIC_DATAFILE_NAME , dict_numeric_data_point_mean)
                            save_dictionary_csv(data_point_path + STRING_DATAFILE_NAME, dict_str_info_data_point_mean)
                            #Saving numeric data to a file with data from each data point.
                            save_dictionary_csv(sweep_folder_path + COLLECTIVE_NUMERIC_DATA_FILENAME, dict_numeric_data_point_mean)
                                
                            print('Iterations complete')
                            
                            
                            
                            
                            
                            
                            
                                
                                
                        



    
    
