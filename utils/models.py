""" @PETER HALLDESTAM, 2020

    models to analyze
    
"""

from tensorflow.keras import Model, Input, regularizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Concatenate
import tensorflow.keras.backend as K

from utils.layers import GraphConv
from utils.layers import res_net_block
from utils.layers import non_res_block
from utils.layers import dense_res_net_block

from utils.tensors import get_adjacency_matrix

def FCN(no_inputs, no_outputs, no_layers, no_nodes):
    """
    Args:
        no_inputs  : number of input nodes
        no_outputs : number of ouput nodes
        no_layers  : number of fully-connected layers
        no_nodes   : number of nodes in each layer
    Returns:
        fully-connected neural network as tensorflow.keras.Model object
    """
    inputs = Input(shape=(no_inputs,), dtype='float32')
    
    if no_layers==0:
        outputs = Dense(no_outputs, activation='linear')(inputs)
    
    else:
        x = Dense(no_nodes, activation='relu')(inputs)
        for i in range(no_layers-1):
            x = Dense(no_nodes, activation='relu')(x)
        outputs = Dense(no_outputs, activation='linear')(x)
        
    model = Model(inputs, outputs)
    model._name = 'FCN'
    return model

def regularized_FCN(no_inputs, no_outputs, no_layers, no_nodes, l1=0.01, l2=0.01):
    """
    Args:
        no_inputs  : number of input nodes
        no_outputs : number of ouput nodes
        no_layers  : number of fully-connected layers
        no_nodes   : number of nodes in each layer
    Returns:
        fully-connected neural network that applies weight regualization 
        as tensorflow.keras.Model object 
    """
    inputs = Input(shape=(no_inputs,), dtype='float32')
    
    if no_layers==0:
        outputs = Dense(no_outputs, activation='linear')(inputs)
    
    else:
        x = Dense(no_nodes,  kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                  activation='relu')(inputs)
        for i in range(no_layers-1):
            x = Dense(no_nodes, 
                      kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                      activation='relu')(x)
        outputs = Dense(no_outputs, activation='linear')(x)
        
    model = Model(inputs, outputs)
    model._name = 'regularized_FCN'
    return model
            


def GCN(no_inputs, no_outputs, no_layers, no_nodes):
    
    inputs = Input(shape=(no_inputs,), dtype='float32')
    
    x = GraphConv(no_inputs, activation='relu')(inputs)
    x = GraphConv(no_inputs, activation='relu')(x)
    
    x = Dense(no_nodes, activation='relu')(inputs)
    for i in range(no_layers-2):
        x = Dense(no_nodes, activation='relu')(x)
    
    outputs = Dense(no_outputs, activation='linear')(x)
    
    model = Model(inputs, outputs)
    model._name = 'GCN'
    return model

def ResNet(no_inputs, no_outputs, blocks):
    
    inputs = Input(shape=(no_inputs,), dtype='float32')
    
    x = res_net_block(inputs)
    for i in range(blocks):
        x = res_net_block(x)
    x = res_net_block(x)
    
    outputs = outputs = Dense(no_outputs, activation='linear')(x)
    
    model = Model(inputs, outputs)
    model._name = 'ResNet'
    return model


def Dense_ResNet(no_inputs, no_outputs, no_nodes, no_blocks, no_skipped_layers):
    """    

    Parameters
    ----------
     no_inputs  : number of input nodes
     no_outputs : number of ouput nodes
     no_nodes   : number of nodes in each Dense layer
     no_blocks  : number of Dense ResNet blocks
     no_skipped_layers : number of layers to be skipped for each resnet block

    Returns
    -------
    model : TYPE
        DESCRIPTION.

    """
    
    
    inputs = Input(shape=(no_inputs,), dtype='float32')
    
    if no_blocks==0:
        outputs = Dense(no_outputs, activation='linear')(inputs)
    
    else:
        x = Dense(no_nodes, activation='relu')(inputs)
        for i in range(no_blocks-1):
           x = dense_res_net_block(x, no_nodes, no_skipped_layers)
        outputs = Dense(no_outputs, activation='linear')(x)
        
    model = Model(inputs, outputs)
    model._name = 'Dense_ResNet'
    
    return model


#classification, need more work
def FCN_(no_inputs, no_outputs, no_layers, no_nodes,
        cartesian=False, classification=False):
    """
    Args:
        no_inputs  : number of input nodes
        no_outputs : number of ouput nodes
        no_layers  : number of fully-connected layers
        no_nodes   : number of nodes in each layer
        cartesian_coordinates : matters only with classification nodes
        classifcation_nodes : True if training with classification nodes
    Returns:
        fully-connected neural network as tensorflow.keras.Model object
    """
    inputs = Input(shape=(no_inputs,), dtype='float32')
    x = Dense(no_nodes, activation='relu', use_bias=False)(inputs)
    for i in range(no_layers-2):
        x = Dense(no_nodes, activation='relu')(x)
        
    if classification:
        no_classifications = int(no_outputs/4)
        no_regression = no_outputs-no_classifications
    
        output1 = Dense(no_regression, activation='linear')(x)                 #for regression
        output2 = Dense(no_classifications, activation='sigmoid')(x)           #for classification
        outputs = Concatenate(axis=1)([output1, output2])
        
    else:
        outputs = Dense(no_outputs, activation='linear')(x)
        
    return Model(inputs, outputs)

    
