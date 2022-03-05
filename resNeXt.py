import os
from datetime import timedelta
from time import time
import numpy as np

from tensorflow.keras.optimizers import Adam
from tensorflow.distribute import MirroredStrategy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from loss_function.loss import LossFunction

from utils.plot_methods import plot_predictions_bar
from utils.plot_methods import plot_loss

from utils.data_preprocess import load_data
from utils.data_preprocess import get_eval_data

from utils.help_methods import save
from utils.help_methods import save_summary
from utils.help_methods import save_dictionary_csv
from utils.help_methods import check_folder
from utils.help_methods import get_measurement_of_performance
from utils.help_methods import get_permutation_match
from utils.help_methods import cartesian_to_spherical

from utils.models import ResNeXtDense

## ----------- Load and prepare data for training/evaluation -------------------

NPZ_DATAFILE = os.path.join('../data', 'XB_mixed_e6_m3_clus50.npz')

TOTAL_PORTION = 0.3      #portion of file data to be used, (0,1]
EVAL_PORTION = 0.2      #portion of total data for final evalutation (0,1)
VALIDATION_SPLIT = 0.1  #portion of training data for epoch validation

#load simulation data. OBS. labels need to be ordered in decreasing energy!
data, labels = load_data(NPZ_DATAFILE, TOTAL_PORTION)

#detach subset for final evaluation, train and eval are inputs, train_ and eval_ are labels
train, train_, eval, eval_ = get_eval_data(data, labels, eval_portion=EVAL_PORTION)
max_mult = int(len(train_[0])/3)

## ---------------------- Build the neural network -----------------------------
# initiate the network structure
 
get_folder = check_folder('Training_ResNeXt/training')    

my_ckpts = get_folder + "/cp-{epoch:04d}.ckpt"
checkpoint_callback = ModelCheckpoint(
    filepath=my_ckpts,
    monitor='val_loss',
    verbose=1,
    save_best_only=False,
    save_weights_only=True,
    mode='auto',
    save_freq='epoch',
    options=None
)

units = 128
cardinality = 4

strategy = MirroredStrategy()
with strategy.scope(): ## for multi gpu single node use
    model = ResNeXtDense(units=units,cardinality=cardinality)

# select mean squared error as loss function
#loss = LossFunction(max_mult, regression_loss='squared')

#compile the network
LEARNING_RATE = 1e-4    # learning rate/step size
loss = LossFunction(max_mult, regression_loss='squared')
optimizer = Adam(lr=LEARNING_RATE)
model.compile(optimizer=Adam(lr=LEARNING_RATE), loss=loss.get(), metrics=['accuracy'])

## ----------------- Train the neural network and plot results -----------------

#train the network with training data
NO_EPOCHS = 250         # no. times to go through training data
BATCH_SIZE = 1024       # the training batch size
start_time = time()
training = model.fit(train, train_,
                     epochs=NO_EPOCHS,
                     batch_size=BATCH_SIZE,
                     validation_split=VALIDATION_SPLIT,
                     callbacks=[EarlyStopping(monitor='val_loss', patience=3), checkpoint_callback])
ttime = time() - start_time
# plot the learning curve

learning_curve = plot_loss(training)

# get predictions on evaluation data
predictions = model.predict(eval)

# return the combination that minimized the loss function (out of max_mult! possible combinations)

predictions, eval_ = get_permutation_match(predictions, eval_, loss, max_mult)

predictions = cartesian_to_spherical(predictions, error=True)
eval_ = cartesian_to_spherical(eval_, error=True)
figure, rec_events = plot_predictions_bar(predictions, eval_, show_detector_angles=True)

meas_perf = get_measurement_of_performance(predictions, eval_, spherical=True)
nn_info = {'Loss': training.history['loss'][-1], 'P mean': meas_perf['momentum mean'], 'P std': meas_perf['momentum std'],
        'Batch size': BATCH_SIZE, 'Units': units, 'Cardinality': cardinality, 'Learning rate': '{:.2e}'.format(LEARNING_RATE),
        'Epochs': [training.epoch[-1] + 1, NO_EPOCHS], 'Network': model._name, 
        'Optimizer': [optimizer._name, float(optimizer.decay), float(optimizer.beta_1),float(optimizer.beta_2)],
        'Training time': timedelta(seconds=round(ttime)), 'Data file': NPZ_DATAFILE,
        'Number of events': len(data[:,0])}

# save figures, parameters etc
save(get_folder + '/model', figure, learning_curve, model)
save_dictionary_csv(get_folder + '/model/info.csv', nn_info )
save_summary(get_folder + '/model', model)
model.save(get_folder + '/model/saved_model')
