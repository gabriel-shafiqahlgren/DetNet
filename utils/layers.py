""" @PETER HALLDESTAM, 2020

    network layer to implement
    
"""
import numpy as np
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras import activations
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.nn import relu
from tensorflow import concat
from tensorflow import split

from utils.tensors import get_adjacency_matrix


class GraphConv(Layer):
    def __init__(self, no_outputs, activation=None):
        self.no_outputs = no_outputs
        self.activation = activations.get(activation)
        super().__init__()
        
        #create the normalized adjacency matrix
        adj = get_adjacency_matrix() + np.eye(162)
        self.adj_norm = K.constant(adj)
        d = np.diag(np.power(np.array(adj.sum(1)), -0.5).flatten())
        self.adj_norm = K.constant(adj.dot(d).transpose().dot(d))
        
        
		
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.no_outputs),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super().build(input_shape)
        
    def call(self, x):
        conv = K.dot(self.adj_norm, K.transpose(x))
        output = K.dot(K.transpose(conv), self.kernel)
        return self.activation(output)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.no_outputs)
        

def res_net_block(input_data):
    x = GraphConv(162, activation=None)(input_data)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = GraphConv(162, activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Add()([x, input_data])
    x = Activation('relu')(x)
    return x


def non_res_block(input_data, filters, conv_size):
    x = GraphConv(162, activation=None)(input_data)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = GraphConv(162, activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def dense_res_net_block(x, no_nodes, no_skipped_layers):
    if no_skipped_layers == 0:
       output = Dense(no_nodes, activation='relu')(x)
    else:
        y = Dense(no_nodes, activation='relu')(x)
        for i in range(no_skipped_layers-1):
            y = Dense(no_nodes, activation='relu')(y)
        output = Add()([x, y])
    return output

class GroupDense(Layer):
    def __init__(self, 
                 units,         
                 activation=None,
                 groups=1,
                 group_depth=1,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(GroupDense, self).__init__()

        self.units = units
        self.activation = activation
        self.groups = groups
        self.group_depth = group_depth # Number of layers in one group, between split and concant
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        self.dense_list = [[]] * self.groups # Defines nested list of layers (matrix) 
        for i in range(self.groups):
            for j in range(self.group_depth):
                self.dense_list[i].append(Dense(units=units,
                                            activation=activation,
                                            use_bias=use_bias,
                                            kernel_initializer=kernel_initializer,
                                            bias_initializer=bias_initializer,
                                            kernel_regularizer=kernel_regularizer,
                                            bias_regularizer=bias_regularizer,
                                            activity_regularizer=activity_regularizer,
                                            kernel_constraint=kernel_constraint,
                                            bias_constraint=bias_constraint,
                                            **kwargs))
        self.dense_list = np.array(self.dense_list)

    def call(self, inputs, **kwargs):
        feature_map_list = []
        for i in range(self.groups):
            x_i = self.dense_list[i][0](split(inputs,self.groups,axis=1)[i])
            for j in range(1,self.group_depth):
                x_i = self.dense_list[i,j](x_i)
            feature_map_list.append(x_i)
        out = concat(feature_map_list, axis=-1)
        return out


class ResNeXt_BottleNeck(Layer):
    def __init__(self, units, groups, depth, group_depth):
        super(ResNeXt_BottleNeck, self).__init__()
        self.depth = depth # depth: Number of group dense blocks. One group dense block splits tensor for each group, goes through the layers and then concats.
        self.bn = BatchNormalization()
        self.dense1 = Dense(32 * groups, activation='relu')  
        self.group_dense = GroupDense(units=units,
                                      groups=groups,
                                      group_depth=group_depth)
        
        self.group_dense_list = []     
        
        for i in range(self.depth):
            self.group_dense_list.append(GroupDense(units=units,groups=groups,group_depth=group_depth))
            
        self.shortcut_dense = Dense(units=units * groups)
        
    def call(self, inputs, training=None, **kwargs):
        x = self.dense1(inputs)
        for i in range(self.depth):
            x = self.group_dense_list[i](x)
            
#         x = self.bn(x)
        shortcut = self.shortcut_dense(inputs)
        
        output = relu(add([x, shortcut]))
        return output


def build_ResNeXt_block_dense(units, groups, depth, group_depth, repeat_num):
    block = Sequential()
    block.add(ResNeXt_BottleNeck(units=units,
                                 groups=groups,
                                 depth=depth,
                                 group_depth=group_depth))
    for _ in range(1, repeat_num):
        block.add(ResNeXt_BottleNeck(units=units,
                                     groups=groups, 
                                     depth=depth,
                                     group_depth=group_depth))

    return block
