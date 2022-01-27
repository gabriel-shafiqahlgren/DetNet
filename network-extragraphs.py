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

## for extra graphs

import pandas as pd
import scipy.optimize as opt
import datashader as ds
import datashader.transfer_functions as tf
import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
from scipy.stats import pearsonr
from datashader.utils import export_image

## ----------- Load and prepare data for training/evaluation -------------------

NPZ_DATAFILE = os.path.join(os.getcwd(), 'data', 'XB_mixed_data_e7.npz')

TOTAL_PORTION = 1      #portion of file data to be used, (0,1]
EVAL_PORTION = 0.2      #portion of total data for final evalutation (0,1)
VALIDATION_SPLIT = 0.1  #portion of training data for epoch validation

#load simulation data. OBS. labels need to be ordered in decreasing energy!
data, labels = load_data(NPZ_DATAFILE, TOTAL_PORTION)

#detach subset for final evaluation, train and eval are inputs, train_ and eval_ are labels
train, train_, eval, eval_ = get_eval_data(data, labels, eval_portion=EVAL_PORTION)

## ---------------------- Build the neural network -----------------------------

# initiate the network structure
NO_LAYERS = 5
NO_NODES = 124
no_inputs = len(train[0])
no_outputs = len(train_[0])
model = FCN(no_inputs, no_outputs, NO_LAYERS, NO_NODES)

# select mean squared error as loss function
max_mult = int(no_outputs / 3)
loss = LossFunction(max_mult, regression_loss='squared')

#compile the network
LEARNING_RATE = 1e-4    # learning rate/step size
model.compile(optimizer=Adam(lr=LEARNING_RATE), loss=loss.get(), metrics=['accuracy'])

## ----------------- Train the neural network and plot results -----------------

#train the network with training data
NO_EPOCHS = 10          # no. times to go through training data
BATCH_SIZE = 2**9       # the training batch size
training = model.fit(train, train_,
                     epochs=NO_EPOCHS,
                     batch_size=BATCH_SIZE,
                     validation_split=VALIDATION_SPLIT,
                     callbacks=[EarlyStopping(monitor='val_loss', patience=3)])

# plot the learning curve
learning_curve = plot_loss(training)

# get predictions on evaluation data
predictions = model.predict(eval)

# return the combination that minimized the loss function (out of max_mult! possible combinations)
predictions, eval_ = get_permutation_match(predictions, eval_, loss, max_mult)

# plot the "lasersvärd" in spherical coordinates (to compare with previous year)
predictions = cartesian_to_spherical(predictions, error=True)
eval_ = cartesian_to_spherical(eval_, error=True)
figure, rec_events = plot_predictions(predictions, eval_, show_detector_angles=True)

# save figures and trained parameters

Direc = 'ann-graphs'
save(Direc, figure, learning_curve, model)


## ----------- for extra graphs --------------------


# get directory

all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
exmps = [d for d in all_subdirs if Direc in d]
latest_exmp = max(exmps, key=os.path.getmtime)
#
x = lambda n: eval_[:,n] # x(0,1,2) eval of energy,theta,phi -- simulated / 'correct'
y = lambda n: predictions[:,n] # ANN generated predictions

# least squares for E , theta, phi

def func(x,A): #function to curve fit, want A to be ~1
    return x*A
## cut off too high E values because for some reason prediction
## of E is capped at 20 MeV when evals can be higher.
E = 10
#Energy limit for least squares (MeV),
#set E to equal event_recon graph. Only affects curve fit, covariance
list_pred=list(y(0)) # Y
list_eval_=list(x(0)) # X

i=0
while i<len(list_pred): ## any predicted or correct E pair of values
                        ## (for E> something)gets deleted from array
    if list_pred[i] > E or list_eval_[i] > E:
        del list_eval_[i]
        del list_pred[i]
    else:
        i+=1
####
w_E, cov_E = opt.curve_fit(func, list_eval_,list_pred,  p0=1)
w_T, cov_T = opt.curve_fit(func, x(1), y(1), p0=1)
w_P, cov_P = opt.curve_fit(func, x(2), y(2), p0=1)
corr_E, _ = pearsonr(list_eval_,list_pred)
corr_T, _ = pearsonr(x(1), y(1))
corr_P, _ = pearsonr(x(2), y(2))


tmpstr= ["E slope:" + str(w_E) + ". E cov: " + str(cov_E), "Theta slope:"+ str(w_T) +\
         ". Theta covariance:" + str(cov_T), "Phi slope:" + str(w_P) + ". Phi cov:" + str(cov_P),\
         "E Pearson cor: " + str(corr_E), "Theta Pearson cor: " + str(corr_T), "Phi Pearson cor: " + str(corr_P)]
np.savetxt(latest_exmp+"/lsq.txt", tmpstr, fmt='%s') #Curve fit txt file,

## For the 3 seperate graphs


##

df0 = pd.DataFrame(data=np.column_stack((x(0),y(0))),columns=['colX','colY']) #Energy
df1 = pd.DataFrame(data=np.column_stack((x(1),y(1))),columns=['colX','colY']) #Theta
df2 = pd.DataFrame(data=np.column_stack((x(2),y(2))),columns=['colX','colY']) #Phi


N = 450 # Square plot width/height px
E = 10 #np.max(eval_[:,0])*0.75 # X,Y range of axis,in MeV
E_min = np.min(eval_[:,0])

i0 = tf.Image(tf.shade(ds.Canvas(plot_width=N, plot_height=N, x_range=(0,E), y_range=(0,E),
                   x_axis_type='linear', y_axis_type='linear').points(df0,'colX','colY'), cmap=cc.fire))

i1 = tf.Image(tf.shade(ds.Canvas(plot_width=N, plot_height=N, x_range=(0,3.1415), y_range=(0,3.1415),
                   x_axis_type='linear', y_axis_type='linear').points(df1,'colX','colY'), cmap=cc.fire))

i2 = tf.Image(tf.shade(ds.Canvas(plot_width=N, plot_height=N, x_range=(0,2*3.1415), y_range=(0,2*3.1415),
                   x_axis_type='linear', y_axis_type='linear').points(df2,'colX','colY'), cmap=cc.fire))

dftmp = pd.DataFrame(data=np.column_stack((np.linspace(0,E,1000),np.linspace(0,E,1000))),columns=['x','y']) ##
line = tf.Image(tf.shade(ds.Canvas(plot_width=N, plot_height=N, x_range=(0,E), y_range=(0,E),               ##
                   x_axis_type='linear', y_axis_type='linear').points(dftmp,'x','y')))                      ##



export_image(tf.stack(i0,line, name="E"),filename= latest_exmp +'/img-E-' + \
             str(np.round(E_min)) + '-'+ str(np.round(E))+"MeV", background="black")
export_image(tf.stack(i1,line, name="Theta"),filename=latest_exmp +'/img-Theta', background="black")
export_image(tf.stack(i2,line, name="Phi"),filename=latest_exmp +'/img-Phi', background="black")
## ------ 2nd part of graphs -------------
#np.max(x(0))*0.75
max0 = 10  #chosen to match covariance range and event_recon range
max1 = 3.1415
max2 = 3.1415 * 2

fig, axs = plt.subplots(ncols=3, sharey=False, figsize=(25, 7))
fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)


ax = axs[0]
hb = ax.hexbin(x(0), y(0), gridsize=50, bins='log', cmap='inferno')
ax.axis([0, max0, 0, max0])
ax.set_title("E")
cb = fig.colorbar(hb, ax=ax)
cb.set_label('log10(N)')

ax = axs[1]
hb = ax.hexbin(x(1), y(1), gridsize=50, bins='log', cmap='inferno')
ax.axis([0, max1, 0, max1])
ax.set_title("Theta")
cb = fig.colorbar(hb, ax=ax)
cb.set_label('log10(N)')

ax = axs[2]
hb = ax.hexbin(x(2), y(2), gridsize=50, bins='log', cmap='inferno')
ax.axis([0, max2, 0, max2])
ax.set_title("Phi")
cb = fig.colorbar(hb, ax=ax)
cb.set_label('log10(N)')

plt.savefig(latest_exmp+"/img-binsETP.png")