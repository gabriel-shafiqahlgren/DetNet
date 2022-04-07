import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from datetime import timedelta, datetime
from time import gmtime, strftime, time
from numpy import mean, std, ones, zeros, array, ceil, linspace
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from loss_function.loss import LossFunction
from utils.data_preprocess import load_data, get_eval_data
from utils.models import ResNeXtHybrid
from utils.plot_methods import plot_momentum_error_dist, plot_predictions_bar, plot_loss, plot_predictions
from utils.help_methods import get_mem_use, get_gpu_memory
from utils.help_methods import save_dictionary_csv, save_summary, save_figs
from utils.help_methods import get_permutation_match, get_measurement_of_performance
from utils.help_methods import cartesian_to_spherical
from utils.help_methods import check_folder

TOTAL_PORTION = 1  # portion of file data to be used, (0,1]
VALIDATION_SPLIT = 0.1  # portion of training data for epoch validation

## ----------- Load and prepare data for training/evaluation -------------------

NPZ_PATH = '/cephyr/NOBACKUP/groups/snic2022-5-74/DetNet/data'
NPZ_DATAFILE = os.path.join(NPZ_PATH, '3maxmul_0.1_10MeV_3000000_clus300.npz')
NPZ_DATAFILE_EVAL = os.path.join(NPZ_PATH, '3maxmul_0.1_10MeV_500000_clus300.npz')

train, train_ = load_data(NPZ_DATAFILE, total_portion=TOTAL_PORTION, portion_zeros=0.05)
eval, eval_ = load_data(NPZ_DATAFILE_EVAL, total_portion=1, portion_zeros=0.05)

def units_matrix(l, w, n, kappa):
    # kappa: the neurons of a column is some inital value * kappa to the power of the column.
    # if kappa = 1 and alpha = 1: uniform network.
    alpha = 1 # alpha < 1 the decreases the width with increasing depth for all paths.
    triangle = lambda i, a, b: [ ceil(linspace(a, b, i)[j]) for j in range(i)]    
    matrix_map = [triangle(l[k], ceil(n * kappa**k), ceil(n * kappa**(w-1)) * alpha ) for k in range(w)] # Nested list of neurons.
    units = zeros((l[0],w))
    for i in range(w):
        for j in range(l[i]):
            units[j,i] = matrix_map[i][j]
            
    # for i in range(1,w):   # Sets the last layer of all paths to be equal to the last layer of the first path.
    #     units[l[i]-1,i] = units[l[0]-1,0]
        
    for i in range(w):  # Seths the last layers of all paths to some value.    
        units[l[i]-1,i] = 1600
        
    return units  

path_d = [6,3] # Depth for each path.
units = units_matrix(path_d, len(path_d), 1900, 1)[:,:]
units

NO_EPOCHS = 450  # no. times to go through training data

BATCH_SIZE = 2**10
LEARNING_RATE = 1e-4  # learning rate/step size

group_depth = 1 # Number of GroupDenseHybrid blocks
blocks = 1 # Number of ResNeXtHybrid blocks

beta_1 = 0.9 # def 0.9
beta_2 = 0.999 # def 0.999
epsilon = 1e-7 
patience = 10

regression_loss = 'absolute' # 'squared', 'absolute'
norm_layer = 'weight_norm' # 'weight_norm', 'batch_norm' or 'layer_norm'
training_folder = 'Training_ResNeXt'

dMetrics = {'Loss': [], 'P Std': [], 'P Mean': [], 'Training t': [], 'Epochs': [],
                                  'Memory use': [], 'GPU memory use': [], 'Iterations': 0}

## Here you can change name of folder, subfolder    
training_folder = 'Training_ResNeXt'
get_folder = check_folder(training_folder + '/iterations')

## ---------------------- Build the neural network -----------------------------
# initiate the network structure
model = ResNeXtHybrid(units=units, group_depth=group_depth, blocks=blocks, norm_layer=norm_layer)

# select mean squared error as loss function
max_mult = int(len(train_[0])/3)
loss = LossFunction(max_mult, regression_loss=regression_loss)
# compile the network


decaying_lr = ExponentialDecay(
                1e-4,
                decay_steps=4399 * 14,    
                decay_rate=0.96,
                staircase=True)
optimizer = Adam(learning_rate=decaying_lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
model.compile(optimizer=optimizer, loss=loss.get(), metrics=['accuracy'])

## ----------------- Train the neural network and plot results -----------------
# train the network with training data
#train the network with training data

start_time = time()
training = model.fit(train, train_,
                     epochs=NO_EPOCHS,
                     batch_size=BATCH_SIZE,
                     validation_split=VALIDATION_SPLIT,
                     callbacks=[EarlyStopping(monitor='val_loss', patience=patience)])
ttime = time() - start_time

learning_curve = plot_loss(training)

eval, eval_ = load_data(NPZ_DATAFILE_EVAL, total_portion=1, portion_zeros=0.05)

# get predictions on evaluation data
predictions = model.predict(eval)

# return the combination that minimized the loss function (out of max_mult! possible combinations)
predictions, eval_ = get_permutation_match(predictions, eval_, loss, max_mult)

# plot the "lasersvärd" in spherical coordinates (to compare with previous year)
predictions = cartesian_to_spherical(predictions, error=True)
eval_ = cartesian_to_spherical(eval_, error=True)
rec_fig, rec_events = plot_predictions_bar(predictions, eval_, epsilon=0.01 , show_detector_angles=True)
meas_perf = get_measurement_of_performance(predictions, eval_, spherical=True)

print('P mean', meas_perf['momentum mean'])

dMetrics['Training t'].append(ttime)
dMetrics['P Mean'].append(meas_perf['momentum mean'])
dMetrics['P Std'].append(meas_perf['momentum std'])
dMetrics['Loss'].append(training.history['loss'][-1])
dMetrics['Memory use'].append(round(get_mem_use(), 4))
dMetrics['GPU memory use'].append(round(get_gpu_memory(), 4))
dMetrics['Epochs'].append(training.epoch[-1] + 1)           
dMetrics['Iterations'] +=1

current_time = strftime("%Y-%m-%d %H:%M:%S", gmtime()).replace(" ", "-").replace(':',';')
Direc = get_folder + current_time

# mom_fig = plot_momentum_error_dist(predictions, eval_)

path_to_folder = save_figs(Direc, figs=[learning_curve, rec_fig], model=model)
# path_to_folder = save_figs(Direc, figs=[], model=model)

dct_Data = {'Loss': training.history['loss'][-1], 'P mean': meas_perf['momentum mean'],
            'P std': meas_perf['momentum std'] ,'Batch size': BATCH_SIZE,'Units': units,
            'Groups': units.shape[-1], 'Learning rate': '{:.2e}'.format(LEARNING_RATE),
            'Epochs': [training.epoch[-1] + 1, NO_EPOCHS], 'Network': model._name, 
            'Optimizer': [optimizer._name,float(optimizer.beta_1),
                          float(optimizer.beta_2)],'Regression loss': regression_loss,
            'Training time': timedelta(seconds=round(ttime)), 'Data file': NPZ_DATAFILE_EVAL,
            'Number of events': train.shape[0], 'Memory GB': round(get_mem_use(), 4),
            'VRAM GB': round(get_gpu_memory(), 4) }

save_dictionary_csv(os.path.join(path_to_folder, 'data.csv'), dct_Data)
save_summary(get_folder, model)
# Saves MME and path to folder in a general csv file for fast comparison amongst runs.
save_dictionary_csv('./' + training_folder +  '/data.csv', {'MME': meas_perf['momentum mean'], 
                                                            'Loss': training.history['loss'][-1], 'Folder': get_folder })