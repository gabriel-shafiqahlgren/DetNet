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

from pandas import DataFrame
from datashader import Canvas
from datashader.transfer_functions import stack
from datashader.transfer_functions import shade
from datashader.transfer_functions import Image
from datashader.utils import export_image
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc

## ----------- Load and prepare data for training/evaluation -------------------
datafile = 'XB_e6_iso_0_10MeV_gamma_numcl10.npz'
NPZ_DATAFILE = os.path.join(os.getcwd(), 'data', datafile)

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
NO_EPOCHS = 10         # no. times to go through training data
BATCH_SIZE = 2**10       # the training batch size
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

# plot the "lasersv???rd" in spherical coordinates (to compare with previous year)
predictions = cartesian_to_spherical(predictions, error=True)
eval_ = cartesian_to_spherical(eval_, error=True)
figure, rec_events = plot_predictions(predictions, eval_, show_detector_angles=True)

# save figures and trained parameters

Direc = 'ann-graphs' 
save(Direc, figure, learning_curve, model)


## ----------- for extra graphs --------------------
  
# get the most recently created 'Direc' directory 
all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
exmps = [d for d in all_subdirs if Direc in d]
recent_dir = max(exmps, key=os.path.getmtime)
#

# Sometimes eval_ and predictions gets split up into multiple arrays, needs to 
# concat them before graphing them
N = int(len(eval_[0,:])/3) #amount of times it split

# x(n) 'correct', eval data: 0->E, 1->Theta, 2->Phi.
# y(n) 'reconstructed', predicted data: 0->E, 1->Theta, 2->Phi.


def x_eval(n): 
    X = [None]*N
    for i in range(N):
        X[i] = eval_[:,n + 3* i]
    return np.concatenate([i for i in X])

def y_pred(n):
    Y = [None]*N
    for i in range(N):
        Y[i] = predictions[:,n + 3* i]
    return np.concatenate([i for i in Y])

# least squares for E, theta, phi

def func(x,A): #function to curve fit, want A to be ~1
    return x*A 
## cut off too high E values
E_max = np.max(x_eval(0))
E_min = min(i for i in x_eval(0) if i > 0)

list_pred=list(y_pred(0)) # Y
list_eval_=list(x_eval(0)) # X

i=0                     ## Makes it so maximum of x and y are the same
while i<len(list_pred): ## any predicted or correct E pair of values 
                        ## (for E> something)gets deleted from array
    if list_pred[i] > E_max or list_eval_[i] > E_max:        
        del list_eval_[i]
        del list_pred[i]
    else:
        i+=1   
        
####       
w_E, _ = curve_fit(func, list_eval_,list_pred,  p0=1) 
w_T, _ = curve_fit(func, x_eval(1), y_pred(1), p0=1)
w_P, _ = curve_fit(func, x_eval(2), y_pred(2), p0=1) 

corr_E, _ = pearsonr(list_eval_,list_pred) 
corr_T, _ = pearsonr(x_eval(1), y_pred(1)) 
corr_P, _ = pearsonr(x_eval(2), y_pred(2))

tmpstr= ["Datafile used: "+ datafile + ", E min: " + str(E_min) +\
        ", E max: " + str(E_max), "E slope:" + str(w_E) ,\
        "Theta slope:"+ str(w_T), "Phi slope:" + str(w_P),"E Pearson cor: " +\
        str(corr_E),"Theta Pearson cor: " + str(corr_T), "Phi Pearson cor: " +\
        str(corr_P)]
print(tmpstr)
np.savetxt(recent_dir+"/info.txt", tmpstr, fmt='%s') #data info txt file,

## For the 3 seperate graphs 

##

N_px = 400 # Square plot width/height px
max0 = [E_max, np.pi, 2*np.pi]

df_l = [DataFrame(data=np.column_stack((x_eval(i),y_pred(i))),columns=['colX','colY']) for i in [0,1,2]]

i_l = [Image(shade(Canvas(plot_width=N_px, plot_height=N_px, x_range=(0,max0[i]), y_range=(0,max0[i]),
                   x_axis_type='linear', y_axis_type='linear').points(df_l[i],'colX','colY'), cmap=cc.fire)) for i in [0,1,2]]


dftmp = DataFrame(data=np.column_stack((np.linspace(0,E_max,1000),np.linspace(0,E_max,1000))),columns=['x','y']) ##
line = Image(shade(Canvas(plot_width=N_px, plot_height=N_px, x_range=(0,E_max), y_range=(0,E_max),               ##
                   x_axis_type='linear', y_axis_type='linear').points(dftmp,'x','y')))                      ##


file_name = ['/img-E-'+str(np.round(E_min)) + '-'+ str(np.round(E_max))+'MeV', '/img-Theta',\
             '/img-Phi' ]
im_name = ['E', 'Theta', 'Phi']

[export_image(stack(i_l[i],line, name=im_name[i]),filename=recent_dir +file_name[i],\
              background="white") for i in [0,1,2] ]

## --------The three hexagonal bin graphs-------------

plot_name = ["Energy: " + str(np.round(E_min)) + " - " + str(np.round(E_max)), "Theta", "Phi"]

fig, axs = plt.subplots(ncols=3, sharey=False, figsize=(25, 7))
fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)

for i in range(3):
    ax = axs[i]
    hb = ax.hexbin(x_eval(i), y_pred(i), gridsize=50, bins='log', cmap='inferno')
    ax.axis([0, max0[i], 0, max0[i]])
    ax.set_title(plot_name[i])
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('log10(N)')

plt.savefig(recent_dir+"/img-binsETP.png")
