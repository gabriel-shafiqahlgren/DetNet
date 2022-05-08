""" @PETER HALLDESTAM, 2020

    models to analyze
    
"""

from tensorflow.keras import Model, Input, regularizers
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import BatchNormalization, Dropout, Add
from tensorflow.keras.layers import GlobalAveragePooling1D
import tensorflow.keras.backend as K

import os

from utils.layers import GraphConv
from utils.layers import res_net_block
from utils.layers import non_res_block
from utils.layers import dense_res_net_block
from utils.layers import ResNeXt_block, ResNeXt_hybrid_block
from utils.layers import convolution_max_pool_block, convolution_max_pool_res_block


from utils.tensors import get_adjacency_matrix

import numpy as np
import tensorflow as tf


def CNN3(no_inputs, no_outputs, depth=1, width=1000, filters=[16, 32, 64],
         conv_blocks=2, block_filters=[128, 256], block_layers=2,
         sort = 'CCT', batch_normalization = True, pooling_before_FCN=True):
    assert(len(block_filters) == conv_blocks)
    """
    Parameters
    ----------
    no_inputs : int
        number of input nodes.
    no_outputs : int
        number of ouput nodes.
    depth : int, optional
    width : int, optional.

    Returns
    -------
    TYPE
        DESCRIPTION.
    """
    
    NEIGHBORS_A, NEIGHBORS_D = 16, 19
    #rotations
    no_rotations_A, no_rotations_D = 5, 6
    rot = "_rot"
    #reflections
    refl = "_refl"

    MAT_PATH = os.path.dirname(__file__) + '/../ConvolutionalMatrix/'
    print(MAT_PATH)    
    #MAT_PATH = '.../ConvolutionalMatrix/'
    
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
            
    x_A = Conv1D(filters[1], kernel_size=no_rotations_A, strides=no_rotations_A, activation='relu')(x_A)
    x_D = Conv1D(filters[1], kernel_size=no_rotations_D, strides=no_rotations_D, activation='relu')(x_D)

    if batch_normalization:
        x_A = BatchNormalization()(x_A)
        x_D = BatchNormalization()(x_D)
        
    x_A = Conv1D(filters[2], kernel_size=2, strides=2, activation='relu')(x_A)
    x_D = Conv1D(filters[2], kernel_size=2, strides=2, activation='relu')(x_D)

    for i in range(conv_blocks):
        x_A = convolution_max_pool_block(x_A, block_filters[i], block_layers, batch_normalization=batch_normalization)    
        x_D = convolution_max_pool_block(x_D, block_filters[i], block_layers, batch_normalization=batch_normalization)

    if pooling_before_FCN:
        x_A = GlobalAveragePooling1D()(x_A)
        x_D = GlobalAveragePooling1D()(x_D)
    else:
        x_A = Flatten()(x_A)
        x_D = Flatten()(x_D)
    
    FCN_in = Concatenate(axis=1)([x_A, x_D])
    
    x = Dense(width, activation='relu')(FCN_in)
    

    for i in range(depth-1):
        x = Dense(width, activation='relu')(x)
        
    outputs = Dense(no_outputs, activation='linear')(x)
    
    return Model(inputs, outputs)


def CNN3_residual(no_inputs, no_outputs, depth=1, width=1000, filters=[16, 32, 64],
         conv_blocks=3, block_filters=[128, 256, 512], block_layers=2,
         sort = 'CCT', batch_normalization = True, pooling_before_FCN=True):

    assert(len(block_filters) == conv_blocks)
    NEIGHBORS_A, NEIGHBORS_D = 16, 19
    
    #rotations
    no_rotations_A, no_rotations_D = 5, 6
    rot = "_rot"
    #reflections
    refl = "_refl"
    
    MAT_PATH = os.path.dirname(__file__) + '/../ConvolutionalMatrix/'
    print(MAT_PATH)
    #MAT_PATH = '.../ConvolutionalMatrix/'
    
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

    res_A = x_A
    res_D = x_D
    #prepare for addition
    res_A = Conv1D(filters[2], kernel_size=1, strides=no_rotations_A*2, activation='relu')(res_A)
    res_D = Conv1D(filters[2], kernel_size=1, strides=no_rotations_D*2, activation='relu')(res_D)

    if batch_normalization:
        x_A = BatchNormalization()(x_A)
        x_D = BatchNormalization()(x_D)
            
    x_A = Conv1D(filters[1], kernel_size=no_rotations_A, strides=no_rotations_A, activation='relu')(x_A)
    x_D = Conv1D(filters[1], kernel_size=no_rotations_D, strides=no_rotations_D, activation='relu')(x_D)
    
    if batch_normalization:
        x_A = BatchNormalization()(x_A)
        x_D = BatchNormalization()(x_D)
        
    x_A = Conv1D(filters[2], kernel_size=2, strides=2, activation='relu')(x_A)
    x_D = Conv1D(filters[2], kernel_size=2, strides=2, activation='relu')(x_D)

    x_A = Add()([x_A, res_A])
    x_D = Add()([x_D, res_D])

    
    for i in range(conv_blocks):
        x_A = convolution_max_pool_res_block(x_A, block_filters[i], block_layers, batch_normalization=batch_normalization)    
        x_D = convolution_max_pool_res_block(x_D, block_filters[i], block_layers, batch_normalization=batch_normalization)
    
    if pooling_before_FCN:
        x_A = GlobalAveragePooling1D()(x_A)
        x_D = GlobalAveragePooling1D()(x_D)
    else:
        x_A = Flatten()(x_A)
        x_D = Flatten()(x_D)
    
    FCN_in = Concatenate(axis=1)([x_A, x_D])
    
    x = Dense(width, activation='relu')(FCN_in)
    
    for i in range(depth-1):
        x = Dense(width, activation='relu')(x)

    outputs = Dense(no_outputs, activation='linear')(x)
    
    return Model(inputs, outputs)



def CNN4_residual(no_inputs, no_outputs, depth=1, width=1000, filters=[16],
         conv_blocks=3, block_filters=[128, 256, 512], block_layers=2,
         sort = 'CCT', batch_normalization = True, pooling_before_FCN=True):

    assert(len(block_filters) == conv_blocks)
    NEIGHBORS_A, NEIGHBORS_D = 16, 19
    
    #rotations
    no_rotations_A, no_rotations_D = 5, 6
    rot = "_rot"
    #reflections
    refl = "_refl"
    
    MAT_PATH = os.path.dirname(__file__) + '/../ConvolutionalMatrix/'
    print(MAT_PATH)
    #MAT_PATH = '.../ConvolutionalMatrix/'
    
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
    
    for i in range(conv_blocks):
        x_A = convolution_max_pool_res_block(x_A, block_filters[i], block_layers, batch_normalization=batch_normalization)    
        x_D = convolution_max_pool_res_block(x_D, block_filters[i], block_layers, batch_normalization=batch_normalization)
    
    if pooling_before_FCN:
        x_A = GlobalAveragePooling1D()(x_A)
        x_D = GlobalAveragePooling1D()(x_D)
    else:
        x_A = Flatten()(x_A)
        x_D = Flatten()(x_D)
    
    FCN_in = Concatenate(axis=1)([x_A, x_D])
    
    x = Dense(width, activation='relu')(FCN_in)
    
    for i in range(depth-1):
        x = Dense(width, activation='relu')(x)

    outputs = Dense(no_outputs, activation='linear')(x)
    
    return Model(inputs, outputs)
