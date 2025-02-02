""" @PETER HALLDESTAM, 2020

    methods used to pre-process the data/label sets
    
"""
import numpy as np
from numpy.random import permutation
import random
from random import sample

from .help_methods import spherical_to_cartesian

from tensorflow.keras.utils import Sequence

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) # Hide stupid warning


class DataGenerator(Sequence):
    def __init__(self, det_data, labels, batch_size):
        self.x, self.y = det_data, labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

def load_data(npz_file, total_portion, portions=None,  add_zeros=0, portion_zeros=0.,
              randomize=True, classification=False, hinge=False, seed=None):
    """
    Reads a .npz-file from the address string: npz_file that contains simulation 
    data in spherical coordinates. I transforms to cartesian coordinates and
    may add n empty events or make some portion of the events be zero-events
    and also add classification nodes.
    
    """
    
    if not total_portion > 0 and total_portion <= 1:
        raise ValueError('total_portion must be in the interval (0,1].')
        
    if isinstance(npz_file,list):
        data_set_list = [np.load(file, mmap_mode='r') for file in npz_file]        
        det_data_list = [data['detector_data'] for data in data_set_list]        
        labels_list = [spherical_to_cartesian(data['energy_labels']) for data in data_set_list]
        if isinstance(portions, list):
            no_events_list = [int(len(labels_list[i])*portions[i]) for i in range(len(portions))] 
            
            det_data_list = [det_data_list[i][:no_events_list[i]] for i in range(len(portions))]
            labels_list = [labels_list[i][:no_events_list[i]] for i in range(len(portions))]
            
        det_data = np.concatenate((det_data_list), axis=0)
        labels = np.concatenate((labels_list), axis=0) 
            
    else:
        data_set = np.load(npz_file, mmap_mode='r')
        det_data = data_set['detector_data']
        labels = spherical_to_cartesian(data_set['energy_labels'])
        
    no_events = len(labels)    
    
    if add_zeros and randomize or 0. < portion_zeros < 1. and randomize:
        if add_zeros:            
            n=add_zeros
            
        elif 0. < portion_zeros < 1.:
            n = int((no_events*portion_zeros)/(1-portion_zeros))  
        
        new_data = list(det_data)
        new_labels = list(labels)

        empty_data = np.zeros(det_data[0].shape)
        empty_label = np.zeros(labels[0].shape)
        
        print('Inserting {} empty events.'.format(n))
        for _ in range(n):
            new_data.append(empty_data)
            new_labels.append(empty_label)
             
        det_data, labels = randomize_data(new_data, new_labels, seed=seed)
        
    elif randomize:
        det_data, labels = randomize_data(det_data, labels, seed=seed) 
        
    elif add_zeros:
        det_data, labels = insert_empty_events(det_data, labels, n=add_zeros)
        
    elif 0. < portion_zeros < 1.:
        n = (no_events*portion_zeros)/(1-portion_zeros)
        det_data, labels = insert_empty_events(det_data, labels, n=int(n))
    
    if classification:
        labels = insert_classification_labels(labels, hinge)
    
    no_events = int(len(labels)*total_portion)
    print('Using {} events from {}'.format(no_events, npz_file))
    return det_data[:no_events], labels[:no_events]


def randomize_data(data, labels, seed = None):
    data, labels = list(data), list(labels) # Needs to be lists of arrays, otherwise will shuffle in wrong axis.
    print('Shuffling data.')
    combined = list(zip(data, labels))
    
    if isinstance(seed, int):
        random.seed(seed)
        
    random.shuffle(combined)
    data[:], labels[:] = zip(*combined)
    return np.array(data), np.array(labels)


def insert_empty_events(data, labels, n):
    """
    Inserts n empty events into given data and label set.
    
    """
    if n > 10*len(labels):
        raise ValueError('Too many zeros to add...')
    
    print('inserting {} zero events...'.format(n))
    
    new_data = list(data)
    new_labels = list(labels)

    empty_data = np.zeros(data[0].shape)
    empty_label = np.zeros(labels[0].shape)

    insert_indices = np.sort(permutation(len(new_labels))[:n]) # Get random indices for insertion
    for i in insert_indices:
        new_data.insert(i, empty_data)
        new_labels.insert(i, empty_label)

    return new_data, new_labels
    

def insert_classification_labels(labels, hinge):
    """
    Inserts binary classification labels (b,px,py,pz) for each event as:
        energy==0 => b=1
        energy>0 => b=0
    """
    px_ = labels[::,0::3].flatten()
    py_ = labels[::,1::3].flatten()
    pz_ = labels[::,2::3].flatten()
    
    energy = np.sqrt(px_*px_+py_*py_+pz_*pz_)
    b_ = np.array((energy!=0)*1)
    
    if hinge:
        b_ = 2*(b_ - .5)
    
    max_mult = int(len(labels[0])/3)
    return np.reshape(np.column_stack((b_, px_, py_, pz_)), (-1, 4*max_mult))
   
    
def get_eval_data(data, labels, eval_portion=0.1):
    """
    Seperates into data and labels set into one for training/validiation and
    another for final evaluation.

    """
    if not eval_portion > 0 and eval_portion < 1:
        raise ValueError('eval_portion must be in interval (0,1).')
    no_eval = int(len(data)*eval_portion)
    return data[no_eval:], labels[no_eval:], data[:no_eval], labels[:no_eval]


def sort_data(old_npz, new_npz):
    """
    Sort the label data matrix with rows (energy1, theta1, phi1, ... phiM), 
    where M is the max multiplicity and saves to new npz file.

    """
    data_set = np.load(old_npz)
    labels = data_set['energy_labels']
    max_mult = int(len(labels[0])/3)
    energies = labels[::,::3]
    sort_indices = np.argsort(-energies) 
    sorted_labels = np.zeros([len(labels), 3*max_mult])
    for i in range(len(labels)):
        for j in range(max_mult):
            sorted_labels[i,3*j:3*(j+1)] = labels[i,3*sort_indices[i,j]:3*(sort_indices[i,j]+1)]
    np.savez(new_npz, detector_data=data_set['detector_data'], energy_labels=sorted_labels)
