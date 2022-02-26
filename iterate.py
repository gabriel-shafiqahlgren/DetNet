import os
from datetime import timedelta, datetime
from time import gmtime, strftime, time
from numpy import mean, std

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from loss_function.loss import LossFunction

from utils.data_preprocess import load_data
from utils.data_preprocess import get_eval_data

from utils.models import FCN

from utils.help_methods import get_mem_use
from utils.help_methods import get_gpu_memory
from utils.help_methods import save_dictionary_csv
from utils.help_methods import save_figs
from utils.help_methods import get_permutation_match
from utils.help_methods import get_measurement_of_performance
from utils.help_methods import cartesian_to_spherical

'''
Trains the network a ceratain number of times and saves the weights and parameters of every
network. 
'''


## ----------- Load and prepare data for training/evaluation -------------------
datafile = 'XB_mixed_e6_m3_clus50.npz'
NPZ_DATAFILE = os.path.join('./data', datafile)

TOTAL_PORTION = 1  # portion of file data to be used, (0,1]
EVAL_PORTION = 0.2  # portion of total data for final evalutation (0,1)
VALIDATION_SPLIT = 0.1  # portion of training data for epoch validation

NO_EPOCHS = 5  # 200        # no. times to go through training data
BATCH_SIZE = 20000  # the training batch size
LEARNING_RATE = 1e-4  # learning rate/step size
NO_LAYERS = 7  # 10
NO_NODES = 50  # 1000
beta_1 = 0.9985
beta_2 = 0.99999

patience = 4


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

def network(iterations):
    dMetrics: Dict[str, List[Any]] = {'Loss': [], 'P Std': [], 'P Mean': [], 'Training t': [], 'Epochs': [],
                                      'Memory use': [], 'GPU memory use': [], 'Iterations': 0}
    
    ## Here you can change name of folder, subfolder    
    get_folder = check_folder('ann_iterations/iterations')
    
    def get_metrics(): #Inner function that analyses the data from the for loop.
        minMax = lambda x: [round(min(x), 3), round(mean(x), 3), round(max(x), 3)]
        mean_loss = round(mean(dMetrics['Loss']), 4)
        std_loss = round(std(dMetrics['Loss']), 4)
        mean_mme = round(mean(dMetrics['P Mean']), 4)
        std_mme = round(mean(dMetrics['P Std']), 4)
        mean_epochs = round(mean(dMetrics['Epochs']), 4)
        attime = str(timedelta(seconds=round(mean(dMetrics['Training t']))))
        
        number_of_events = len(data[:, 0])

        # Probably wont work if the optimizer is changed from Adam. 
        # Since it needs the .decay, .beta_1, .beta_2 attributes.
        dict_metrics = {'Iterations': dMetrics['Iterations'], 'Mean loss': mean_loss, 'Std loss': std_loss,
                        'Mean of mean p error': mean_mme,
                        'Mean of std p error': std_mme, 'Batch size': BATCH_SIZE, 'Nodes': NO_NODES, 'Layers': NO_LAYERS,
                        'Learning rate': '{:.2e}'.format(LEARNING_RATE), 'Epochs [mean stop, max]': [mean_epochs, NO_EPOCHS],
                        'Patience': patience,
                        'Network': model._name,
                        'Optimizer': [optimizer._name, float(optimizer.decay), float(optimizer.beta_1),
                                   float(optimizer.beta_2)],
                        'Regression loss': regression_loss, 'Average training time': attime, 'Data file': datafile, 'Save folder': get_folder,
                        'Number of events': number_of_events, 'Memory GB [min, mean, max]': minMax(dMetrics['Memory use']),
                        'VRAM GB [min, mean, max]': minMax(dMetrics['GPU memory use'])}
        return dict_metrics
    
    try:
        for i in range(iterations):
            print('------------ Iteration: ' + str(i+1) + '/' + str(iterations) + ' ------------')
            # load simulation data. OBS. labels need to be ordered in decreasing energy!
            data, labels = load_data(NPZ_DATAFILE, TOTAL_PORTION)
            # detach subset for final evaluation, train and eval are inputs, train_ and eval_ are labels
            train, train_, eval, eval_ = get_eval_data(data, labels, eval_portion=EVAL_PORTION)

            ## ---------------------- Build the neural network -----------------------------
            # initiate the network structure
            no_inputs = len(train[0])
            no_outputs = len(train_[0])
            model = FCN(no_inputs, no_outputs, NO_LAYERS, NO_NODES)
            # select mean squared error as loss function
            max_mult = int(no_outputs / 3)
            regression_loss = 'squared'
            loss = LossFunction(max_mult, regression_loss=regression_loss)
            # compile the network
            optimizer = Adam(learning_rate=LEARNING_RATE, beta_1=beta_1, beta_2=beta_2, epsilon=1e-08, amsgrad=True)
            model.compile(optimizer=optimizer, loss=loss.get(), metrics=['accuracy'])

            ## ----------------- Train the neural network and plot results -----------------
            # train the network with training data
            start_time = time()
            training = model.fit(train, train_,
                                 epochs=NO_EPOCHS,
                                 batch_size=BATCH_SIZE,
                                 validation_split=VALIDATION_SPLIT,
                                 verbose=1,
                                 callbacks=[EarlyStopping(monitor='val_loss', patience=patience)])

            ttime = time() - start_time
            dMetrics['Training t'].append(ttime)
            print("Training time> --- %s ---" % timedelta(seconds=round(dMetrics['Training t'][i])))
            # get predictions on evaluation data
            predictions = model.predict(eval)
            # return the combination that minimized the loss function (out of max_mult! possible combinations)
            predictions, eval_ = get_permutation_match(predictions, eval_, loss, max_mult)
            
            predictions = cartesian_to_spherical(predictions, error=True)
            eval_ = cartesian_to_spherical(eval_, error=True)
            meas_perf = get_measurement_of_performance(predictions, eval_, spherical=True)

            dMetrics['P Mean'].append(meas_perf['momentum mean'])
            dMetrics['P Std'].append(meas_perf['momentum std'])
            dMetrics['Loss'].append(training.history['loss'][-1])
            dMetrics['Memory use'].append(round(get_mem_use(), 4))
            dMetrics['GPU memory use'].append(round(get_gpu_memory(), 4))
            dMetrics['Epochs'].append(training.epoch[-1] + 1)           
            dMetrics['Iterations'] +=1
            
            current_time = strftime("%Y-%m-%d %H:%M:%S", gmtime()).replace(" ", "-").replace(':',';')
            Direc = get_folder + current_time
            path_to_folder = save_figs(Direc, figs=[], model=model)
            
            dct_Data = {'Loss': training.history['loss'][-1], 'P mean': meas_perf['momentum mean'], 'P std': meas_perf['momentum std'] ,'Batch size': BATCH_SIZE, 
                            'Nodes': NO_NODES, 'Layers': NO_LAYERS, 'Learning rate': '{:.2e}'.format(LEARNING_RATE),
                            'Epochs': [training.epoch[-1] + 1, NO_EPOCHS], 'Network': model._name, 
                            'Optimizer': [optimizer._name, float(optimizer.decay), float(optimizer.beta_1),float(optimizer.beta_2)],
                            'Regression loss': regression_loss, 'Training time': timedelta(seconds=round(ttime)), 'Data file': datafile,
                            'Number of events': len(data[:,0]), 'Memory GB': round(get_mem_use(), 4),
                            'VRAM GB': round(get_gpu_memory(), 4) }
            save_dictionary_csv(os.path.join(path_to_folder, 'data.csv'), dct_Data)
            
    except KeyboardInterrupt: # Makes it so you can CTRL-C to stop the program and save the already done iterations.
        return get_metrics()
    return get_metrics()

n = 5 # Number of times to train the network.
dictionary = network(n)
folder = dictionary['Save folder']
CSVfilename = folder + 'datasum.csv'
save_dictionary_csv(CSVfilename, dictionary)
