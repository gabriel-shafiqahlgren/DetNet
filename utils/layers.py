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

class ListDense(Layer):
    '''
    A 'rectangular' basic FCN used in GroupDense.
    '''
    def __init__(self, 
                 units=100,    
                 list_depth=5):
        super(ListDense, self).__init__()    
        
        self.units = units
        self.list_depth = list_depth        
        self.layers = [Dense(units, activation='relu') for i in range(list_depth)]
        
    def call(self, inputs):
        x = self.layers[0](inputs)
        for i in range(1,self.list_depth):
            x = self.layers[i](x)
            
        return x

class GroupDense(Layer):
    '''
    Splits input tensor to a number (cardinality/group) of 'ListDense' layers
    and then concats the output of each group.
    '''
    def __init__(self, 
                 units=100,         
                 groups=3,
                 list_depth=3):
        super(GroupDense, self).__init__()

        self.units = units
        self.groups = groups
        self.list_depth = list_depth
        
        self.listDense = [ListDense(units, list_depth) for i in range(groups)]

    def call(self, inputs):
        feature_map_list = []
        for i in range(self.groups):
            x_i = self.listDense[i](split(inputs,self.groups,axis=1)[i])
            feature_map_list.append(x_i)
        out = concat(feature_map_list, axis=-1)
        return out
    
class ResNeXt(Layer):
    '''
    units: Number of neurons in each 'Dense' fully connected layer.
    groups: Number of times to split the input tensor and send each part to a list of cascading dense layers.
    group_depth: Number of 'Group Dense' layers in one ResNeXt block, normally '1' in ResNeXt.
    list_depth: Number of 'Dense' layers in one group (ListDense) in a 'Group Dense' layer.
    
    Important to initialize layers in __init__.
    '''
    def __init__(self, units, groups, group_depth, list_depth, batch_norm):
        super(ResNeXt, self).__init__()
        self.group_depth = group_depth
        self.batch_norm = batch_norm
        
        self.bn = BatchNormalization()
        
        self.dense = Dense(32 * groups, activation='relu')  
        
        self.group_dense_list = [GroupDense(units=units,groups=groups,list_depth=list_depth) for i in range(group_depth)]     
            
        self.shortcut_dense = Dense(units=units * groups)
        
    def call(self, inputs, training=None, **kwargs):
        x = self.dense(inputs)
        for i in range(self.group_depth):
            x = self.group_dense_list[i](x)
            
        if self.batch_norm:            
            x = self.bn(x) 
            
        shortcut = self.shortcut_dense(inputs)
        
        output = relu(add([x, shortcut]))
        return output


def ResNeXt_block(units, groups, group_depth, list_depth, repeat_num, batch_norm):
    block = Sequential()
    block.add(ResNeXt(units=units,
                     groups=groups,
                     group_depth=group_depth,
                     list_depth=list_depth,
                     batch_norm=batch_norm))
    for _ in range(1, repeat_num):
        block.add(ResNeXt(units=units,
                         groups=groups, 
                         group_depth=group_depth,
                         list_depth=list_depth,
                         batch_norm=batch_norm))

    return block

class ListDenseHybrid(Layer):
    '''
    Input units> vector.
    '''
    def __init__(self,  
                 list_depth, # Int 
                 units): # Vector length of list_depth
        super(ListDenseHybrid, self).__init__()    
        
        self.list_depth = list_depth 
        self.units = units     
        
        self.layers = [Dense(units[i], activation='relu') for i in range(list_depth)]
        
    def call(self, inputs):
        x = self.layers[0](inputs)
        for i in range(1,self.list_depth):
            x = self.layers[i](x)
            
        return x

class GroupDenseHybrid(Layer):
    '''
    Splits input tensor to a number (cardinality/group) of 'ListDense' layers
    and then concats the output of each group.
    Now involves shortcuts (resnet).
    Assumes that the last amount neurons for every group is the same and that
    the first group is one of the deepest groups.
    '''
    def __init__(self, 
                 units, # Needs matrix as input with size = groups x max(list_depth)
                 groups, # Integer
                 list_depth): # Vector of length = groups
        super(GroupDenseHybrid, self).__init__()

        self.units = units # Matrix
        self.groups = groups # Int
        self.list_depth = list_depth # Vector 
        
        self.shortcut_dense = Dense(units[list_depth[0]-1, 0]) 
        self.list_dense = [ListDenseHybrid(list_depth[i], units[:,i]) for i in range(groups)]

    def call(self, inputs):
        feature_map_list = []
        for i in range(self.groups):
            x_i = self.list_dense[i](split(inputs,self.groups,axis=1)[i])
            shortcut = self.shortcut_dense(split(inputs,self.groups,axis=1)[i])
            output = relu(add([x_i, shortcut]))
            feature_map_list.append(output)
        out = concat(feature_map_list, axis=-1)
        return out

class ResNeXtHybrid(Layer):
    '''
    units: Matrix of neurons for each group, depth
    '''
    def __init__(self, units, group_depth, batch_norm):
        super(ResNeXtHybrid, self).__init__()
        
        # Pads the 'units' matrix so that it always is a square matrix.
        
        n,m = units.shape
        pad1,pad2 = 0,0
        if n > m: 
            pad1 = n-m
            pad2 = 0

        if n < m: 
            pad1 = 0
            pad2 = m-n

        # npad is a tuple of (n_before, n_after) for each dimension
        npad = ((0, pad2), (0, pad1))
        
        self.units = np.pad(units, pad_width=npad, mode='constant', constant_values=0)
        
        # Get the depth of each column. A column can end with zeros.
        list_depth = []

        for i in range(self.units.shape[0]):
            for j in range(self.units.shape[1]):
                if self.units[j,i] == 0:
                    list_depth.append(j)
                    break
                elif j == self.units.shape[0] - 1:
                    list_depth.append(self.units.shape[0])
                    
        list_depth = np.array(list_depth)
        
        self.group_depth = group_depth
        self.batch_norm = batch_norm
        self.groups = len(list_depth)
        
        self.bn = BatchNormalization()
        
        self.dense = Dense(self.units[list_depth[0]-1,0] * self.groups, activation='relu')  
        
        self.group_dense_list = [GroupDenseHybrid(units=self.units,groups=self.groups,list_depth=list_depth) for i in range(group_depth)]     
                    
    def call(self, inputs, training=None, **kwargs):
        x0 = self.dense(inputs)
        x = self.group_dense_list[0](x0)
        for i in range(1,self.group_depth):
            x = self.group_dense_list[i](x)
            
        if self.batch_norm:            
            x = self.bn(x)    
            
        output = relu(add([x, x0]))
        return output
    
def ResNeXt_hybrid_block(units, group_depth, repeat_num, batch_norm):
    block = Sequential()
    block.add(ResNeXtHybrid(units=units,
                     group_depth=group_depth,
                     batch_norm=batch_norm))
    for _ in range(1, repeat_num):
        block.add(ResNeXtHybrid(units=units,
                         group_depth=group_depth,
                         batch_norm=batch_norm))

    return block