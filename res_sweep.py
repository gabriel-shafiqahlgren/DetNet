import os
import sys
from tensorflow.keras.optimizers import Adam
from utils.custom_callbacks import EarlyStoppingAtMinLoss

from tensorflow.keras.optimizers.schedules import CosineDecay, ExponentialDecay
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.distribute import MirroredStrategy
from tensorflow.distribute.cluster_resolver import SlurmClusterResolver

from tensorflow.train import latest_checkpoint

from utils.models import ResNeXtHybrid

from loss_function.loss import LossFunction

from utils.data_preprocess import load_data, randomize_data, DataGenerator
from utils.help_methods import get_no_trainable_parameters
from utils.help_methods import check_folder
from utils.help_methods import get_permutation_match
from utils.help_methods import get_measurement_of_performance

from time import gmtime, strftime, time
from datetime import timedelta, datetime
from utils.help_methods import get_mem_use
from utils.help_methods import get_gpu_memory
from utils.help_methods import save_dictionary_csv
from utils.help_methods import save_figs
from utils.help_methods import save_summary
from numpy import mean, std, linspace, logspace, sort, arange, ceil, zeros, flip, log10

def units_matrix(l, n, kappa, alpha):
    w = len(l)
    # kappa: the neurons of a column is some inital value * kappa to the power of the column.
    # if kappa = 1 and alpha = 1: uniform network.
    triangle = lambda i, a, b: [ ceil(linspace(a, b, i)[j]) for j in range(i)]    
    matrix_map = [triangle(l[k], ceil(n * kappa**k), ceil(n * kappa**(w-1)) * alpha ) for k in range(w)] # Nested list of neurons.
    units = zeros((l[0],w))
    for i in range(w):
        for j in range(l[i]):
            units[j,i] = matrix_map[i][j]
            
    for i in range(1,w):   # Sets the last layer of all paths to be equal to the last layer of the first path.
        units[l[i]-1,i] = units[l[0]-1,0]
        
    return units  

## --------------- Constants
VALIDATION_SPLIT = 0.1  #portion of training data for epoch validation
SWEEP_DATA_FOLDER = 'resnext_v2'
NUMERIC_DATAFILE_NAME = 'numeric_data.csv'
STRING_DATAFILE_NAME = 'info.csv'
LOAD_FILENAME_EVAL = 'eval10_train.npz' #Constant file for testing
PORTION_ZEROS = 0.05 #portion of loaded data to be made into zero events
COLLECTIVE_NUMERIC_DATA_FILENAME = 'data_matrix.csv'
LOAD_FILE_NAME = 'training_big_data_all.npz'
# npz_datafile = os.path.join('/cephyr/NOBACKUP/groups/snic2022-5-74/DetNet/data', '3maxmul_0.1_10MeV_3000000_clus300.npz')
npz_datafile = '/cephyr/NOBACKUP/groups/snic2022-5-74/training/' +  LOAD_FILE_NAME
ITERATIONS_PER_DATA_POINT = 1
SWEEP_NAME = 'portion_full_test'

## --------------- Sweep parameters
ls_total_port = [1] #portion of file data to be used, (0,1]
ls_units = [units_matrix([5,5], 1800, 1, 1)]
ls_depth = [1]
ls_group_depth = [1]
ls_no_epochs = [80]
ls_no_batch_size = [2**10]
ls_learning_rate = [1e-4]
ls_blocks = [1]
ls_decay_steps = [5000]

patience = 6
norm_layer = 'none'
regression_loss = 'absolute'

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

def get_metrics(dMetrics, batch_size, units, group_depth, blocks, lr_rate, decay_steps, no_epochs, load_filename, sweep_folder_path, data, no_parameters):
        
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
                                    'Units': units, 
                                    'Group depth': group_depth,
                                    'Blocks': blocks,
                                    'Normalization layer': norm_layer,
                                    'Learning rate': '{:.2e}'.format(lr_rate), 
                                    'Decay steps': decay_steps,
                                    'Epochs mean': mean_epochs, 'Epochs stop':no_epochs, 'Epochs max':max_epochs, 
                                    'Patience': patience, 
                                    'Average training time': avg_training_time_min,
                                    'Trainable parameters': no_parameters,
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
                                     'Optimizer': [optimizer._name], 
                                     'Regression loss': regression_loss,
                                     'Data file': load_filename, }
    return dict_numeric_data_point_mean, dict_str_info_data_point_mean



#load_filename = input_data_filename()
load_filename = LOAD_FILE_NAME

sweep_folder_path = input_sweep_folder_name(load_filename, SWEEP_NAME)

#iterations = input_number_of_iterations()
iterations = ITERATIONS_PER_DATA_POINT

# Hope ypu have enough RAM

npz_datafile_eval = os.path.join('/cephyr/NOBACKUP/groups/snic2022-5-74/eval', LOAD_FILENAME_EVAL)
npz_datafile_val = os.path.join('/cephyr/NOBACKUP/groups/snic2022-5-74/eval', LOAD_FILENAME_EVAL)

for total_port in ls_total_port:
    #load simulation data for training. OBS. labels need to be ordered in decreasing energy!
    
    #loat simulation data for evaluation. -/-
    eval, eval_ = load_data(npz_datafile_eval, total_portion=1, portion_zeros=PORTION_ZEROS)
    val, val_ = load_data(npz_datafile_val, total_portion=1, portion_zeros=PORTION_ZEROS)    
    train, train_ = load_data(npz_datafile, total_portion=total_port, portions = 1, portion_zeros=PORTION_ZEROS)       
    
    
    for decay_steps in ls_decay_steps:
        for units in ls_units:
            for no_epochs in ls_no_epochs:
                for batch_size in ls_no_batch_size:
                    for lr_rate in ls_learning_rate:
                        for group_depth in ls_group_depth:
                            for blocks in ls_blocks:
                                
                                train_gen = DataGenerator(train, train_, batch_size)
                                val_gen = DataGenerator(val, val_, batch_size)
                                
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

                                    len_data = len(train_)
                                    n = int(len_data/iterations)
                                    ## ---------------------- Build the neural network -----------------------------

                                    # Derived parameters
                                    no_inputs = len(train[0])
                                    no_outputs = len(train_[0])
                                    
                                    strategy = MirroredStrategy()
                                    with strategy.scope(): ## for multi gpu single node use
                                        model = ResNeXtHybrid(units=units,
                                                          group_depth=group_depth, 
                                                          blocks=blocks,
                                                          norm_layer=norm_layer)

                                    decaying_lr = ExponentialDecay(
                                                    lr_rate,
                                                    decay_steps=decay_steps,    
                                                    decay_rate=0.96,
                                                    staircase=True)

#                                         decaying_lr = CosineDecay(
#                                                         lr_rate,
#                                                         decay_steps = decay_steps,
#                                                         alpha = cos_alpha)

                                    max_mult = int(no_outputs / 3)
                                    # select mean squared error as loss function
                                    loss = LossFunction(max_mult, regression_loss=regression_loss)

                                    optimizer=Adam(learning_rate=decaying_lr)

                                    #compile the network
                                    model.compile(optimizer, loss=loss.get(), metrics=['accuracy'])

                                    ## ----------------- Train the neural network and plot results -----------------
                                    start_time = time()
                                    training = model.fit(train_gen,
                                                         epochs=no_epochs,
                                                         batch_size=batch_size,
                                                         validation_data=val_gen,
                                                         callbacks=[EarlyStoppingAtMinLoss(patience=patience)])
                                    training_time = time() - start_time

                                    dMetrics['Training time'].append(training_time)
                                    print("Training time> --- %s ---" % timedelta(seconds=round(dMetrics['Training time'][i])))

                                    # get predictions on evaluation data
                                    predictions = model.predict(eval)

                                    # return the combination that minimized the loss function (out of max_mult! possible combinations)
                                    predictions, eval_ = get_permutation_match(predictions, eval_, loss, max_mult)

                                    # print the error in E, theta and phi a
                                    meas_perf = get_measurement_of_performance(predictions, eval_, spherical=False)

                                    no_parameters = get_no_trainable_parameters(model) 

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
                                    path_to_folder = save_figs(iteration_path, figs=[], model=None)

                                    #data saved collectively for each iteration
                                    dct_Data = {'Loss': training.history['loss'][-1], 
                                                'P mean': meas_perf['momentum mean'], 
                                                'P std': meas_perf['momentum std'] ,
                                                'Batch size': batch_size, 
                                                'Units': units, 
                                                'Group blocks': group_depth,
                                                'ResNeXt Blocks': blocks,
                                                'Decay steps': decay_steps,
                                                'Learning rate': '{:.2e}'.format(lr_rate),
                                                'Epochs': [training.epoch[-1] + 1, no_epochs], 
                                                'Network': model._name, 
                                                'Optimizer': [optimizer._name],
                                                'Regression loss': regression_loss, 
                                                'Training time': timedelta(seconds=round(training_time)), 
                                                'Data file': load_filename,
                                                'Number of events': len(train[:,0]), 
                                                'Memory GB': round(get_mem_use(), 4),
                                                'VRAM GB': round(get_gpu_memory(), 4) 
                                                }

                                    save_dictionary_csv(os.path.join(path_to_folder, 'data.csv'), dct_Data)

                                dict_numeric_data_point_mean, dict_str_info_data_point_mean = \
                                    get_metrics(dMetrics, batch_size, units, group_depth, blocks, lr_rate, decay_steps,
                                                no_epochs, load_filename, sweep_folder_path,train, no_parameters)
                                #Saving data to data point filder
                                save_dictionary_csv(data_point_path + NUMERIC_DATAFILE_NAME , dict_numeric_data_point_mean)
                                save_dictionary_csv(data_point_path + STRING_DATAFILE_NAME, dict_str_info_data_point_mean)
                                #Saving numeric data to a file with data from each data point.
#                                 model.save(data_point_path+'/model')
                                save_dictionary_csv(sweep_folder_path + COLLECTIVE_NUMERIC_DATA_FILENAME, dict_numeric_data_point_mean)

                                print('Iterations complete')
