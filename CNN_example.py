""" @PETER HALLDESTAM, 2020
    
    Construction and training of a neural network. Easy to make changes to any 
    parameters and most importantly to implements different structures.

"""

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
import sys
import os
import pickle
import time
start = time.time()

from utils.models import CNN

#from loss_funcion.loss_functions import loss_function_wrapper
from loss_function.loss import LossFunction
from utils.data_preprocess import load_data, get_eval_data
from utils.help_methods import get_permutation_match, cartesian_to_spherical, get_measurement_of_performance
from contextlib import redirect_stdout


## ----------------------------- PARAMETERS -----------------------------------

#### C-C-F NETWORK ####

NPZ_DATAFILE = os.path.join(os.getcwd(), 'data', '3maxmul_0.1_10MeV_3000000_clus300.npz')
            #or import sys and use sys.argv[1]
TOTAL_PORTION = 0.3333333333                                #portion of file data to be used, (0,1]
EVAL_PORTION = 0.2                              #portion of total data for final evalutation (0,1)
VALIDATION_SPLIT = 0.1                          #portion of training data for epoch validation
CARTESIAN = True                                #train with cartesian coordinates instead of spherical
CLASSIFICATION = False                          #train with classification nodes

NO_EPOCHS = 500
                                               #Number of times to go through training data
BATCH_SIZE = 2**8                                #The training batch size
LEARNING_RATE = 1e-4                            #Learning rate/step size
PERMUTATION = True                              #set false if using an ordered data set                          #type of loss: {mse, modulo, cosine} (only mse for cartesian)
MAT_SORT = "CCT"                                #type of sorting used for the convolutional matrix
USE_ROTATIONS = True   
USE_REFLECTIONS = True
NAME = 'regular_CNN'


USE_BATCH_NORMALIZATION = True 

FILTERS = [32, 16]    # [32, 16]                #must consist of even numbers!
DEPTH = 3    # 3
                
def main():
    folder = NAME
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
    print("Skapat mapp: ", os.getcwd()+folder)
    
    #load simulation data. OBS. labels need to be ordered in decreasing energy!
    
#load simulation data. OBS. labels need to be ordered in decreasing energy!
    data, labels = load_data(NPZ_DATAFILE, TOTAL_PORTION)
    
    #detach subset for final evaluation. train_** is for both training and validation
    train_data, train_labels, eval_data, eval_labels = get_eval_data(data, labels,
                                                                     eval_portion=EVAL_PORTION)
    
    
    ### ------------- BUILD, TRAIN & TEST THE NEURAL NETWORK ------------------
    
    
    #no. inputs/outputs based on data set
    no_inputs = len(train_data[0])                  
    no_outputs = len(train_labels[0])               
    
    #initiate the network structure

    model = CNN(no_inputs, no_outputs, sort = MAT_SORT, filters = FILTERS,
                depth = DEPTH, 
                rotations = USE_ROTATIONS, reflections = USE_REFLECTIONS,
                batch_normalization = USE_BATCH_NORMALIZATION)
    
    max_mult = int(no_outputs / 3)
    loss = LossFunction(max_mult, regression_loss='squared')

    
    """
    #select loss function
    loss_function = loss_function_wrapper(no_outputs, 
                                          loss_type=LOSS_FUNCTION, 
                                          permutation=PERMUTATION,
                                          cartesian=CARTESIAN,
                                          classification=CLASSIFICATION)
    """
    
    #select optimizer
    opt = Adam(lr=LEARNING_RATE)
   
    #compile the network
    model.compile(optimizer=opt, loss=loss.get(), metrics=['accuracy'])
    
    es = EarlyStopping(monitor='val_loss', patience=5)
    mcp = ModelCheckpoint(filepath=folder+'/checkpoint', monitor='val_loss')
    
    training = model.fit(train_data, train_labels, 
                         epochs=NO_EPOCHS, batch_size=BATCH_SIZE,
                         validation_split=VALIDATION_SPLIT,
                         callbacks=[es, mcp])

    epochs = es.stopped_epoch
    if epochs == 0:
        epocs = NO_EPOCHS
    #plot predictions on evaluation data
    predictions = model.predict(eval_data)

    
    if CARTESIAN:
        predictions = cartesian_to_spherical(predictions)
        eval_labels = cartesian_to_spherical(eval_labels)    
    if PERMUTATION:
        predictions, labels = get_permutation_match(predictions, eval_labels, loss, max_mult)

    
    #save weights
    model.save_weights(folder+'/weights.h5')
    
    #save summary and time
    with open(folder+'/modelsummary.txt', 'w') as f:
        with redirect_stdout(f):
            end = time.time()
            hours, rem = divmod(end-start, 3600)
            print("Elapsed time: ")
            minutes, seconds = divmod(rem, 60)
            print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
            print("Elapsed epochs: ", epochs)
            model.summary()
    
    #save history
    with open(folder+'/traininghistory', 'wb') as file_pi:
        pickle.dump(training.history, file_pi)
    
    #save predicted events and measurement of performance
    y = predictions
    y_ = eval_labels
    events = {'predicted_energy': y[::,0::3].flatten(),
              'correct_energy': y_[::,0::3].flatten(), 
              
              'predicted_theta': y[::,1::3].flatten(),
              'correct_theta': y_[::,1::3].flatten(),
              
              'predicted_phi': np.mod(y[::,2::3], 2*np.pi).flatten(),
              'correct_phi': y_[::,2::3].flatten()}
    np.save(folder+'/events',events)
    mop = get_measurement_of_performance(y, y_)
    np.save(folder+'/mop',mop)
    return

if __name__ == '__main__':
    main()
