""" @PETER HALLDESTAM, 2020

    models to analyze
    
"""

from tensorflow.keras import Model, Input, regularizers
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import BatchNormalization
import tensorflow.keras.backend as K

from utils.layers import GraphConv
from utils.layers import res_net_block
from utils.layers import non_res_block
from utils.layers import dense_res_net_block
from utils.layers import ResNeXt_block, ResNeXt_hybrid_block

from utils.tensors import get_adjacency_matrix

import numpy as np
import tensorflow as tf

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
    
def ResNeXtDense(units=64, cardinality=32, group_depth=1,  list_depth=3, blocks=1, batch_norm=False):
    no_outputs = 9
    no_inputs = 162

    inputs = Input(shape=(no_inputs,), dtype='float32')

    x = Dense(units, activation='relu')(inputs)

    x = ResNeXt_block(units=units,
                      groups=cardinality,
                      group_depth=group_depth,
                      list_depth=list_depth,
                      repeat_num=blocks,
                      batch_norm=batch_norm)(x)

    outputs = Dense(no_outputs, activation='linear')(x)      
    model = Model(inputs, outputs)
    model._name = 'ResNeXtDense'
    return model

def ResNeXtHybrid(units, group_depth = 1, blocks = 1, batch_norm=True):
    no_outputs = 9
    no_inputs = 162
    # units: a matrix, with each column is a group of layers,
    # assuming first column is one of the deepest and all groups have the same
    # amount of neurons in the last layer, if not it wont be able to concat.
    
    # Example for units: units = 500 * np.array([[1,1,1],[1,1,0],[1,0,0]])
    # The groups/paths are the columns of the 'units' matrix.
    # Doesnt have to a square matrix. Cant have a zero in middle of a group, will cause problems.
    
    # 'group_depth' is the number of GroupDense layers.
    # 'blocks' is the number of ResNeXt_hybrid_block.
    
    inputs = Input(shape=(no_inputs,), dtype='float32')

    x = ResNeXt_hybrid_block(units=units,
                      group_depth=group_depth,
                      repeat_num=blocks,
                      batch_norm=batch_norm)(inputs)

    outputs = Dense(no_outputs, activation='linear')(x)      
    model = Model(inputs, outputs)
    model._name = 'ResNeXtHybrid'
    return model


##### Convolutional networks #####


def CNN(no_inputs, no_outputs, depth=3, width=80, filters = [256, 16, 4],
        sort = 'CCT', rotations = True, reflections = True, 
        batch_normalization = True):
    """

    Parameters
    ----------
    no_inputs : int
        number of input nodes.
    no_outputs : int
        number of ouput nodes.
    depth : int, optional
        depth of FCN-bit. The default is 2.
    width : int, optional
        width of FCN-bit. The default is 8.
    pooling_type : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    NEIGHBORS_A, NEIGHBORS_D = 16, 19
    
    
    if rotations:
        no_rotations_A, no_rotations_D = 5, 6
        rot = "_rot"
    else:
        no_rotations_A, no_rotations_D = 1, 1
        rot = ""
    
    if reflections:
        refl = "_refl"
    else:
        refl = ""
    
    MAT_PATH = '../ConvolutionalMatrix/'
    
    A_mat = np.load(MAT_PATH+'A_mat_'+sort+rot+refl+'.npy')
    D_mat = np.load(MAT_PATH+'D_mat_'+sort+rot+refl+'.npy')
    
    inputs = Input(shape=(no_inputs,), dtype='float32')
    
    A_in = tf.matmul(inputs, A_mat)
    D_in = tf.matmul(inputs, D_mat)
    
    #reshapes the input to [batch, steps, channels]
    
    A_in = tf.reshape(A_in, [-1, A_in.shape[1], 1])
    D_in = tf.reshape(D_in, [-1, D_in.shape[1], 1])
   
    #parameters for conv1D: filters, kernel size, stride, activation
    
    x_A = Conv1D(filters[0], NEIGHBORS_A, NEIGHBORS_A, activation='relu', 
                 input_shape = (None, A_in.shape[1], 1), data_format = "channels_last" )(A_in)
    
    x_D = Conv1D(filters[0], NEIGHBORS_D, NEIGHBORS_D, activation='relu', 
                 input_shape = (None, D_in.shape[1], 1), data_format = "channels_last" )(D_in)
    
    if batch_normalization:
        x_A = BatchNormalization()(x_A)
        x_D = BatchNormalization()(x_D)
    
    x_A = Conv1D(filters[1], no_rotations_A, no_rotations_A, activation='relu')(x_A)
    x_D = Conv1D(filters[1], no_rotations_D, no_rotations_D, activation='relu')(x_D)
    
    if batch_normalization:
        x_A = BatchNormalization()(x_A)
        x_D = BatchNormalization()(x_D)
    
    
    x_A = Flatten()(x_A)
    x_D = Flatten()(x_D)
    
    FCN_in = Concatenate(axis=1)([x_A, x_D])
    
    x = Dense(width, activation='relu')(FCN_in)
    

    for i in range(depth-1):
        x = Dense(width, activation='relu')(x)
        
    outputs = Dense(no_outputs, activation='linear')(x)
    
    return Model(inputs, outputs)

def CN_FCN(no_inputs, no_outputs, depth=[3, 3], width=80, filters = [32, 16],
        sort = 'CCT', rotations = True, reflections = True, 
        batch_normalization = True):
    """

    Parameters
    ----------
    no_inputs : int
        number of input nodes.
    no_outputs : int
        number of ouput nodes.
    depth : int, optional
        depth of FCN-bit. The default is 2.
    width : int, optional
        width of FCN-bit. The default is 8.
    pooling_type : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    NEIGHBORS_A, NEIGHBORS_D = 16, 19
    
    
    if rotations:
        no_rotations_A, no_rotations_D = 5, 6
        rot = "_rot"
    else:
        no_rotations_A, no_rotations_D = 1, 1
        rot = ""
    
    if reflections:
        refl = "_refl"
    else:
        refl = ""
    
    MAT_PATH = 'ConvolutionalMatrix/'
    
    A_mat = np.load(MAT_PATH+'A_mat_'+sort+rot+refl+'.npy')
    D_mat = np.load(MAT_PATH+'D_mat_'+sort+rot+refl+'.npy')
    
    inputs = Input(shape=(no_inputs,), dtype='float32')
    
    A_in = tf.matmul(inputs, A_mat)
    D_in = tf.matmul(inputs, D_mat)
    
    #reshapes the input to [batch, steps, channels]
    
    A_in = tf.reshape(A_in, [-1, A_in.shape[1], 1])
    D_in = tf.reshape(D_in, [-1, D_in.shape[1], 1])
   
    #parameters for conv1D: filters, kernel size, stride, activation
    
    x_A = Conv1D(filters[0], NEIGHBORS_A, NEIGHBORS_A, activation='relu', name='Conv_cluster_A',
                 input_shape = (None, A_in.shape[1], 1), data_format = "channels_last" )(A_in)
    
    x_D = Conv1D(filters[0], NEIGHBORS_D, NEIGHBORS_D, activation='relu', name='Conv_cluster_D',
                 input_shape = (None, D_in.shape[1], 1), data_format = "channels_last" )(D_in)
    
    if batch_normalization:
        x_A = BatchNormalization()(x_A)
        x_D = BatchNormalization()(x_D)
        
    #FCN-tjosan
    for i in range(depth[0]):
        x_A = Conv1D(filters[0], 1, 1, activation = 'relu', name = 'FCN_emu_A' + str(i+1))(x_A)
        x_D = Conv1D(filters[0], 1, 1, activation = 'relu', name = 'FCN_emu_D' + str(i+1))(x_D)
    
    x_A = Conv1D(filters[1], no_rotations_A, no_rotations_A, 
                 activation='relu', name = 'Conv_orientation_A')(x_A)
    x_D = Conv1D(filters[1], no_rotations_D, no_rotations_D,
                 activation='relu', name = 'Conv_orientation_D')(x_D)

    if batch_normalization:
        x_A = BatchNormalization()(x_A)
        x_D = BatchNormalization()(x_D)
        
    #x_A = MaxPooling1D(pool_size=2)(x_A)
    #x_D = MaxPooling1D(pool_size=2)(x_D)
    
    x_A = Flatten()(x_A)
    x_D = Flatten()(x_D)
    
    FCN_in = Concatenate(axis=1)([x_A, x_D])
    
    x = Dense(width, activation='relu')(FCN_in)
    
    for i in range(depth[1]-1):
        x = Dense(width, activation='relu')(x)
        
    outputs = Dense(no_outputs, activation='linear')(x)
    
    return Model(inputs, outputs)

