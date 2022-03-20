import os
from datetime import timedelta
from time import time
import numpy as np

from tensorflow.keras.optimizers.experimental import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.train import latest_checkpoint

from loss_function.loss import LossFunction

from utils.plot_methods import plot_predictions_bar
from utils.plot_methods import plot_loss
from utils.plot_methods import plot_momentum_error_dist

from utils.data_preprocess import load_data
from utils.data_preprocess import get_eval_data

from utils.help_methods import save_figs
from utils.help_methods import save_summary
from utils.help_methods import save_dictionary_csv
from utils.help_methods import check_folder
from utils.help_methods import get_measurement_of_performance
from utils.help_methods import get_permutation_match
from utils.help_methods import cartesian_to_spherical

from utils.models import ResNeXtDense

## ----------- Load and prepare data for training/evaluation -------------------

NPZ_DATAFILE = os.path.join('../data', 'XB_mixed_e6_m3_clus50.npz')

TOTAL_PORTION = 1      #portion of file data to be used, (0,1]
EVAL_PORTION = 0.2      #portion of total data for final evalutation (0,1)
VALIDATION_SPLIT = 0.3  #portion of training data for epoch validation

#load simulation data. OBS. labels need to be ordered in decreasing energy!
data, labels = load_data(NPZ_DATAFILE, TOTAL_PORTION)

#detach subset for final evaluation, train and eval are inputs, train_ and eval_ are labels
train, train_, eval, eval_ = get_eval_data(data, labels, eval_portion=EVAL_PORTION)
max_mult = int(len(train_[0])/3)

## ---------------------- Build the neural network -----------------------------
# initiate the network structure

units = 64
cardinality = 4
LEARNING_RATE = 0.5e-3
NO_EPOCHS = 1000         # no. times to go through training data
BATCH_SIZE = 2**11      # the training batch size
patience = 7
# strategy = MirroredStrategy()
# with strategy.scope(): ## for multi gpu single node use
model = ResNeXtDense(units=units,cardinality=cardinality)


# select mean squared error as loss function
#loss = LossFunction(max_mult, regression_loss='squared')

#compile the network
    # learning rate/step size
loss = LossFunction(max_mult, regression_loss='squared')
optimizer = Adam(
    learning_rate=LEARNING_RATE, beta_1=0.99, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    clipnorm=None, clipvalue=None, global_clipnorm=None, use_ema=False,
    ema_momentum=0.99, ema_overwrite_frequency=None, jit_compile=False,
    name='Adam')
model.compile(optimizer=optimizer, loss=loss.get(), metrics=['accuracy'])

## ----------------- Train the neural network and plot results -----------------
get_folder = check_folder('Training_ResNeXt/training')    

my_ckpts = get_folder + "/cp-{epoch:04d}.ckpt"
checkpoint_callback = ModelCheckpoint(
    filepath=my_ckpts,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode='auto',
    save_freq='epoch',
    options=None
)

#train the network with training data

start_time = time()
training = model.fit(train, train_,
                     epochs=NO_EPOCHS,
                     batch_size=BATCH_SIZE,
                     validation_split=VALIDATION_SPLIT,
                     callbacks=[EarlyStopping(monitor='val_loss', patience=patience),
                                checkpoint_callback])
ttime = time() - start_time

# Load the best weights
latest = latest_checkpoint(get_folder)
model.load_weights(latest)

# Remove all but latest check point

index_min = min(range(len(training.history['val_loss'])), key=training.history['val_loss'].__getitem__)

for epoch in range(index_min):
    if os.path.isfile(get_folder + "cp-{epoch:04d}.ckpt.index".format(epoch=epoch+1)):        
        os.remove(get_folder + "cp-{epoch:04d}.ckpt.index".format(epoch=epoch+1))
        os.remove(get_folder + "cp-{epoch:04d}.ckpt.data-00000-of-00001".format(epoch=epoch+1))

# Plot learning curve
learning_curve = plot_loss(training)

# get predictions on evaluation data
predictions = model.predict(eval)

# return the combination that minimized the loss function (out of max_mult! possible combinations)

predictions, eval_ = get_permutation_match(predictions, eval_, loss, max_mult)

predictions = cartesian_to_spherical(predictions, error=True)
eval_ = cartesian_to_spherical(eval_, error=True)
figure, rec_events = plot_predictions_bar(predictions, eval_, show_detector_angles=True)
fig_med = plot_momentum_error_dist(predictions, eval_)

meas_perf = get_measurement_of_performance(predictions, eval_, spherical=True)
nn_info = {'Loss': training.history['loss'][index_min], 'P mean': meas_perf['momentum mean'], 'P std': meas_perf['momentum std'],
        'Batch size': BATCH_SIZE, 'Units': units, 'Cardinality': cardinality, 'Learning rate': '{:.2e}'.format(LEARNING_RATE),
        'Epochs': [training.epoch[-1] + 1, NO_EPOCHS], 'Network': model._name, 
        'Optimizer': [optimizer._name, float(optimizer.decay), float(optimizer.beta_1),float(optimizer.beta_2)],
        'Training time': timedelta(seconds=round(ttime)), 'Data file': NPZ_DATAFILE,
        'Number of events': len(data[:,0])}

# save figures, parameters etc
save_figs(get_folder + '/model',[figure, learning_curve, fig_med], model)
save_dictionary_csv(get_folder + '/model/info.csv', nn_info )
save_summary(get_folder + '/model', model)
model.save(get_folder + '/model/saved_model')
