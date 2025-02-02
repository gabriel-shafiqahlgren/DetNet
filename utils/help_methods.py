""" @PETER HALLDESTAM, 2020

    Help methods used in neural_network.py
    
"""
import os
import numpy as np
import time
import csv
import psutil
import datetime
import subprocess as sp
import time

from tensorflow import transpose
from tensorflow.keras import Model

from tensorflow.keras.backend import dot
from tensorflow.keras.backend import constant
from tensorflow.keras.backend import count_params
from tensorflow.keras.backend import argmin
from tensorflow.keras.backend import sum as Ksum
from tensorflow.python.client import device_lib

from utils.tensors import get_permutation_tensor
from utils.tensors import get_identity_tensor

from contextlib import redirect_stdout

from csv import writer

from itertools import permutations


DET_GEOMETRY = os.path.join(os.getcwd(), 'data', 'geom_xb.txt') 


def get_permutation_match(y, y_, loss_function, max_mult, no_batches=10):
    
    """
    Sorts the predictions with corresponding label as the minimum of a square 
    error loss function. Must be used BEFORE plotting the "lasersvärd".

    """    
    start_time = time.time()
    print('Matching predicted data with correct label permutation. May take a while...')
    
    new_y_ = np.empty((0, len(y[0])))
    loss = loss_function.get(train=False)
    
    no_params = int(len(y_[0])/max_mult)
    permutation_tensor = get_permutation_tensor(max_mult, m=no_params)
    
    prediction_batches = np.array_split(y, no_batches)
    labels_batches = np.array_split(y_, no_batches)
    
    for Y, Y_ in zip(prediction_batches, labels_batches):
        
        tY, tY_ = constant(Y), constant(Y_)
        indices = np.array(argmin(loss(tY, tY_), axis=0))
        tY_ = np.array(transpose(dot(tY_, permutation_tensor), perm=[1,0,2]))
        matched_y_ = [tY_[j, i, ::] for i,j in enumerate(indices)]
        new_y_ = np.append(new_y_, matched_y_, axis=0)
        
    
    print("Permutation time> --- %s seconds ---" % (time.time() - start_time))
    return y, new_y_

def get_permutation_match_with_lists(predictions, labels):
    """
    DEPRICIATED
    
    Sorts the predictions with corresponding label as the minimum of the regular
    eclidean 2-norm. This version does NOT use tensors as done in
    get_permutation_match and instead uses lists
    
    INVARIANT
    Labels and predictions must be of the form
    0,0,0, ... 0,0,0, px1, py1, pz1, px2, py2, pz2m, ... 
    
    Must be used BEFORE plotting the "lasersvärd".
    """
    print('Matching predicted data with correct label permutation. May take a while...')
    
    start_time = time.time()
    
    #Finding the max number of particles in labels and predictions
    maxmult_pred = len(predictions[0,::3])
    maxmult_labels = len(labels[0,::3])
    
    #Lambdas to find extract particle tripplets (px, py, pz)
    get_P_pred = lambda i, j: predictions[i, j:j+3]
    get_P_label = lambda i, k: labels[i, k:k+3]
    
    new_predictions = np.zeros(predictions.shape)
    
    for i in range(predictions.shape[0]):
        #Dict as error value : [j, k] (error of predicted particle j with label praticle k)
        error_index_dict = {}
        available_prediction_j = []
        
        for j in range(maxmult_pred):
            P_pred = get_P_pred(i, j*3) #every third index is new particle 
            
            if np.sum(P_pred**2) != 0: # Match only if not empty prediction (0,0,0)
                available_prediction_j.append(j) #Save possible predicted particles to be matched
                available_label_k = []
                
                for k in range(maxmult_labels):
                    P_label = get_P_label(i, k*3)
                    
                    if np.sum(P_label**2) != 0:
                        available_label_k.append(k) #Save possible labeled particles to be matched to
                        error = np.linalg.norm(P_label - P_pred)
                        error_index_dict[error] = [j, k]
        
        #Get a list with[j, k] sorted by lowest error to highest
        sorted_errors = sorted(error_index_dict.items(), key=lambda x:x[0],reverse=False) 
        
        while len(available_prediction_j) > 0 and len(available_label_k) > 0:
            
            change = sorted_errors[0][1] # get the first element (j, k) that will have lowest error
            
            if change[0] in available_prediction_j and change[1] in available_label_k:
                #Get the new prediction inde
                new_pred_index = maxmult_pred*3 - maxmult_labels*3 + change[1]*3
                new_predictions[i, new_pred_index: new_pred_index + 3] = get_P_pred(i, change[0]*3)
                available_prediction_j.remove(change[0])
                available_label_k.remove(change[1])
                
            del sorted_errors[0]
        
        # Comment out here to remove zero matched events
        for l in range(len(available_prediction_j)):
            new_pred_index = l*3
            new_predictions[i, new_pred_index: new_pred_index + 3] = get_P_pred(i, available_prediction_j[l]*3)
                
    
    print("Permutation time> --- %s seconds ---" % (time.time() - start_time))
    return new_predictions, labels


def get_permutation_match_with_permutations(predictions, labels):
    """
    Sorts the predictions with corresponding label as the minimum of the regular
    eclidean 2-norm. This version does NOT use tensors as done in
    get_permutation_match and instead uses permutations of lists.
    
    INVARIANT
    Labels and predictions must be of the form
    0,0,0, ... 0,0,0, px1, py1, pz1, px2, py2, pz2m, ... 
    
    Must be used BEFORE plotting the "lasersvärd".
    """
    print('Matching predicted data with correct label permutation. May take a while...')
    
    start_time = time.time()
    
    #Finding the max number of particles in labels and predictions
    maxmult_pred = len(predictions[0,::3])
    maxmult_labels = len(labels[0,::3])
    
    #Lambdas to find extract particle tripplets (px, py, pz)
    get_P_pred = lambda i, j: predictions[i, j:j+3]
    get_P_label = lambda i, k: labels[i, k:k+3]
    
    new_predictions = np.zeros(predictions.shape)
    
    for i in range(predictions.shape[0]):
        #Save memory by ignoring zero events
        available_prediction_j = []
        for j in range(maxmult_pred):
            P_pred = get_P_pred(i, j*3) #every third index is new particle 
            if np.sum(P_pred**2) != 0: # Match only if not empty prediction (0,0,0)
                available_prediction_j.append(j) #Save possible predicted particles to be matched
        
        available_label_k = []
        for k in range(maxmult_labels):
            P_label = get_P_label(i, k*3)
            if np.sum(P_label**2) != 0:
                available_label_k.append(k) #Save possible labeled particles to be matched to
        
        pred_mult = len(available_prediction_j)
        label_mult = len(available_label_k)
        
        if pred_mult >= label_mult: # label_mult is the limit
        
            j_permutations = list(permutations(available_prediction_j, label_mult))
            error_dict = {}
            
            for permutation in j_permutations:
                error = 0
                
                for l in range(label_mult):
                    P_pred = get_P_pred(i, permutation[l]*3)
                    P_label = get_P_label(i, available_label_k[l]*3)
                    error += np.linalg.norm(P_pred - P_label)
                not_included = list(set(permutation) ^ set(available_prediction_j))
                
                for particle_index in not_included:
                    error += np.linalg.norm(get_P_pred(i, particle_index*3))
                   
                error_dict[error] = list(permutation) #Is necessary to tuple -> list 
                
            sorted_errors = sorted(error_dict.items(), key=lambda x:x[0],reverse=False) 
            optimal_permutation = sorted_errors[0][1] 
            
            for m in range(label_mult):
                new_pred_index = (maxmult_pred - maxmult_labels + available_label_k[m])*3
                
                new_predictions[i, new_pred_index: new_pred_index + 3] = get_P_pred(i, optimal_permutation[m]*3)
                available_prediction_j.remove(optimal_permutation[m])
            
            for n in range(len(available_prediction_j)):
                new_pred_index = n*3
                new_predictions[i, new_pred_index: new_pred_index + 3] = get_P_pred(i, available_prediction_j[n]*3)
                #available_prediction_j.remove(available_prediction_j[n])

            
            
        elif pred_mult < label_mult:
            
            k_permutations = list(permutations(available_label_k, pred_mult))
            error_dict = {}
            
            for permutation in k_permutations:
                error = 0 
                for l in range(pred_mult):
                    P_pred = get_P_pred(i, available_prediction_j[l]*3)
                    P_label = get_P_label(i, permutation[l]*3)
                    error += np.linalg.norm(P_pred - P_label)
                error_dict[error] = list(permutation) #Is necessary to tuple -> list
            
            sorted_errors = sorted(error_dict.items(), key=lambda x:x[0],reverse=False) 
            optimal_permutation = sorted_errors[0][1]
            
            for m in range(pred_mult):
                new_pred_index = (maxmult_pred - maxmult_labels + optimal_permutation[m])*3
                new_predictions[i, new_pred_index: new_pred_index + 3] = get_P_pred(i, available_prediction_j[m]*3)
                
            
    
    print("Permutation time> --- %s seconds ---" % (time.time() - start_time))
    return new_predictions, labels


def spherical_to_cartesian(spherical):
    """
    Coordinate transform (energy, theta, phi) --> (px, py, pz)
    
    """
    energy = spherical[::,0::3]
    theta = spherical[::,1::3]
    phi = spherical[::,2::3]

    px = np.sin(theta)*np.cos(phi)*energy    
    py = np.sin(theta)*np.sin(phi)*energy
    pz = np.cos(theta)*energy
    
    cartesian = np.zeros(np.shape(spherical))
    cartesian[::,0::3] = px
    cartesian[::,1::3] = py
    cartesian[::,2::3] = pz
    return cartesian

def cartesian_to_spherical(cartesian, error=False, tol=1e-3, low=-1.0, high=-0.1):
    """
    Coordinate transform (px, py, pz) --> (energy, theta, phi). Used for labels 
    and predictions after training.

    """
    px = cartesian[::,0::3]
    py = cartesian[::,1::3]
    pz = cartesian[::,2::3]
    energy = np.sqrt(px*px + py*py + pz*pz)
    
    get_theta = lambda z,r: np.arccos(np.divide(z, r, out=np.ones_like(z), where=r>tol))
    get_phi = lambda y,x: np.arctan2(y,x)
    
    if error:
        zero_to_random = 0
    else:
        zero_to_random = np.random.uniform(low, high, size=np.shape(energy))

    #Where E<tol do first, else do second arg, 
    theta = np.where(energy <tol , 0, get_theta(pz, energy))
    phi = np.where(energy <tol , 0, get_phi(py, px))
    energy = np.where(energy <tol , zero_to_random, energy)
    
    spherical = np.zeros(np.shape(cartesian))
    spherical[::,0::3] = energy
    spherical[::,1::3] = theta
    spherical[::,2::3] = np.mod(phi, 2*np.pi)
    return spherical

def get_detector_angles():
    """
    Returns the angles (theta, phi) for each of 162 crystall detectors.
    
    """
    with open(DET_GEOMETRY) as f:
        lines = f.readlines()
        
    theta, phi = np.zeros((162,)), np.zeros((162,))
    lines = [line.strip() for line in lines]
    for i in range(162):
        s = lines[i].split(',')
        theta[i] = float(s[2])
        phi[i] =  float(s[3])
    return theta*np.pi/180, (phi+180)*np.pi/180
    

def get_no_trainable_parameters(compiled_model):
    """
    Returns the no. trainable parameters of given compiled model.
    
    """
    assert isinstance(compiled_model, Model)
    return np.sum([count_params(w) for w in compiled_model.trainable_weights])


def get_momentum_error_dist(y, y_, spherical=True):
    if spherical:
        y = spherical_to_cartesian(y)
        y_ = spherical_to_cartesian(y_)
        
    P_l = lambda q: y[::,q::3].flatten()
    P_l_ = lambda q: y_[::,q::3].flatten()

    P = np.vstack([P_l(0) - P_l_(0),P_l(1) - P_l_(1), P_l(2)- P_l_(2)])
    P_error = [np.linalg.norm(P[:,i]) for i in range(P.shape[1])]
    return P_error

def MSE(y, y_, spherical=True):
    if spherical:
        y = spherical_to_cartesian(y)
        y_ = spherical_to_cartesian(y_)

    P_l = lambda q: y[::,q::3].flatten()
    P_l_ = lambda q: y_[::,q::3].flatten()


    P = np.vstack([P_l(0) - P_l_(0),P_l(1) - P_l_(1), P_l(2)- P_l_(2)])


    squared_error = [np.dot(P[:,i],P[:,i]) for i in range(P.shape[1])]
    return np.mean(squared_error)

def RMSE(y, y_, spherical=True):
    return np.sqrt(MSE(y, y_, spherical=spherical))


def get_measurement_of_performance(y, y_, spherical=True):
    """
    Returns the mean and standard deviation in the error of predicted energy, 
    theta and phi for given predictions and labels.
    
    """
    
    assert(y.shape == y_.shape)
    max_mult = int(y_.shape[1]/3)
    
    if spherical:
        cart_y = spherical_to_cartesian(y)
        cart_y_ = spherical_to_cartesian(y_)
        
        sph_y = y
        sph_y_ = y_
        
    else: 
        sph_y = cartesian_to_spherical(y)
        sph_y_ = cartesian_to_spherical(y_)
        
        cart_y = y
        cart_y_ = y_
        
    energy_error = sph_y[...,0::3]-sph_y_[...,0::3]
    theta_error = sph_y[...,1::3]-sph_y_[...,1::3]
    phi_diff = np.mod(sph_y[...,2::3]-sph_y_[...,2::3], 2*np.pi)
    phi_error = np.where(phi_diff > np.pi, phi_diff - 2*np.pi, phi_diff)

    mean = (np.mean(energy_error), np.mean(theta_error), np.mean(phi_error))
    std = (np.std(energy_error), np.std(theta_error), np.std(phi_error))
    
    P_error = get_momentum_error_dist(cart_y,cart_y_, spherical=False)
    P_mean = np.mean(P_error)
    P_std = np.std(P_error)

    P_error = np.array(P_error)
    P_err_event = np.sum(P_error.reshape(-1, max_mult), axis=1)
    event_mean = np.mean(P_err_event)
    event_std = np.std(P_err_event)

    return {'mean': mean, 'std': std, 'momentum mean': P_mean, 'momentum std': P_std, 
            'event mean': event_mean, 'event std': event_std}
    
def save(folder, figure, learning_curve, model):
    folder0 = folder
    folder_name_taken = True
    n = 0
    while folder_name_taken:
        n += 1
        try:
            os.makedirs(folder)
            folder_name_taken = False
        except FileExistsError:
            folder = folder0 + str(n)
        if n==20: 
            raise ValueError('change name!')
    folder = folder+'/'
    figure.savefig(folder + 'event_reconstruction.jpg', format='jpg')
    # figure.savefig(folder + 'event_reconstruction.eps', format='eps')
    learning_curve.savefig(folder + 'training_curve.jpg', format='jpg')
    model.save_weights(folder + 'weights.h5')
    
    
def save_figs(folder, figs, model):
    # Saves a list of figures and the model weights and returns the folder (string) the files
    # were saved in.
    folder0 = folder
    folder_name_taken = True
    n = 0
    while folder_name_taken:
        n += 1
        try:
            os.makedirs(folder)
            folder_name_taken = False
        except FileExistsError:
            folder = folder0 + str(n)
        if n==200: 
            raise ValueError('change name!')
    folder = folder+'/'
    i = 0
    for figures in figs:
        figures.savefig(folder +'fig' + str(i) +'.jpg', format='jpg')
        i += 1
        
    if model is not None:
        model.save_weights(folder + 'weights.h5')
    return folder
 
def get_available_gpus():
    # Returns tuple of: (the gpu used and the amount of gpus).
    local_device_protos = device_lib.list_local_devices()
    GPUS_array = [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']
    return GPUS_array[0], len(GPUS_array)

def get_mem_use():
    # Returns the memory used by the python process, I think.
    pid = os.getpid()
    python_process = psutil.Process(pid)
    return python_process.memory_info()[0]/2.**30  # memory use in GB...I think

def get_gpu_memory(): 
    # Returns the dedicated GPU memory usage in GB
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_used_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_used_values = [int(x.split()[0]) for i, x in enumerate(memory_used_info)][0]
    return memory_used_values * 0.001048576 # in GB

def save_dictionary_csv(csvfile, dictn):
    # Saves a dictionary to a CSV file. 
    # If no .csv file exists of the given name a new one will be created
    # with the 'keys' of the dictionary as the first row in .csv file.
    # If the file exists it will only add the 'values' of the dictionary,
    # wont overwrite current values.
    if os.path.isfile(csvfile) == False:    
        with open(csvfile, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(dictn.keys())
    with open(csvfile, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(dictn.values())
        
def check_folder(folder): 
    # Checks if a directory exists
    folder0 = folder
    folder_name_taken = True
    n = 0
    while folder_name_taken:
        folder = folder0 + str(n)
        n += 1
        if os.path.isdir(folder):
            folder = folder0 + str(n)
        else:
            folder_name_taken = False
    return folder+'/'

def save_summary(folder, model):
    # Saves summary of model to a txt file.
    with open(folder + '/modelsummary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()

def eval_empty_events(model, loss):    
    no_inputs = model._nested_inputs[0].shape[0]
    no_outputs = model._nested_outputs[0].shape[0]

    zero_eval = np.zeros((30,no_inputs))
    zero_eval_ = np.zeros((30,no_outputs))
    zero_predictions = model.predict(zero_eval)
    # return the combination that minimized the loss function (out of max_mult! possible combinations)
    zero_predictions, zero_eval_ = get_permutation_match(zero_predictions, zero_eval_, loss, int(no_outputs/3))
    return get_measurement_of_performance(zero_predictions, zero_eval_, spherical=False)
            
