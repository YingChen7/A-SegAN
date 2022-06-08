# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import tensorflow.compat.v1 as tf1

### dense block
def dense_block(x, nb_layers, growth_rate, isTraining, dropout_rate = None):
    concat_x = x
    for i in range(nb_layers):
        # Conv
        x = tf1.layers.conv3d(inputs = concat_x, filters = growth_rate, kernel_size = 3, strides = 1, padding = 'same', use_bias = False)
        # ReLU         
        x = tf.nn.relu(x)
        # BN
        x = tf1.layers.batch_normalization(inputs = x, training = True)
        concat_x  = tf.concat(values = [concat_x, x], axis = -1)
    return concat_x
        
### transition block
def transition_block(x, nb_filters, isTraining, dropout_rate = None):
    # Conv
    x = tf1.layers.conv3d(inputs = x, filters = nb_filters, kernel_size = 1, strides = 1, padding = 'same', use_bias = False)
    # ReLU
    x = tf.nn.relu(x)
    # BN
    x = tf1.layers.batch_normalization(inputs = x, training = True)
    mid_x  = x
    # Max Pooling
    x = tf1.layers.max_pooling3d(inputs = x, pool_size = 2, strides = 2, padding = 'same')
    return x, mid_x

def attention_block(features_high, features_low):
    gamma = tf.Variable(tf.zeros([1]))
    high_b, high_h, high_w, high_d, high_c = features_high.get_shape().as_list()
    low_b, low_h, low_w, low_d, low_c = features_low.get_shape().as_list()
    features_high_t = tf.transpose(features_high, [0,4,1,2,3])
    features_high_t = tf.reshape(features_high_t, shape = [high_b, high_c, high_h * high_w * high_d])
    features_low_t = tf.reshape(features_low, shape = [low_b, low_h * low_w * low_d, low_c])
    
    attention_mat = tf.linalg.matmul(features_high_t, features_low_t)
    attention_vec = tf1.reduce_mean(tf1.reduce_sum(attention_mat, axis = 1), axis = 0)
    attention_vec = tf.math.sigmoid(attention_vec)
    
    features_low_f = gamma * tf.multiply(features_low, attention_vec) + features_low

    return tf.concat(values = [features_high, features_low_f], axis = -1)

       
def A_Dense_UNet(x, nb_classes, isTraining, dropout_rate, reuse = False):
    print('creating A-Dense-U-Net ...')
    with tf1.variable_scope('A_Dense_UNet') as scope:
        if reuse:
            scope.reuse_variables() 
        with tf1.variable_scope('Pre-operation') as scope:    
            ## Conv-ReLU-BN-Dropout       
            x = tf1.layers.conv3d(inputs = x, filters = 24, kernel_size = 3, strides = 1, padding = 'same', use_bias = False)
            x = tf.nn.relu(x)
            x = tf1.layers.batch_normalization(inputs = x, training = True)
                
        with tf1.variable_scope('Dense_Block_1'):    
            x = dense_block(x, nb_layers = 6, growth_rate = 12, isTraining = isTraining, dropout_rate = dropout_rate)       # 16 + 72 = 88 (64*64*64)
        with tf1.variable_scope('Transition_Block_1'):
            x, mid_x1 = transition_block(x, nb_filters = 96, isTraining = isTraining, dropout_rate = dropout_rate)
            
        with tf1.variable_scope('Dense_Block_2'):
            x = dense_block(x, nb_layers = 6, growth_rate = 14, isTraining = isTraining, dropout_rate = dropout_rate)       # 88 + 84 = 172 (32*32*32)
        with tf1.variable_scope('Transition_Block_2'):
            x, mid_x2 = transition_block(x, nb_filters = 180, isTraining = isTraining, dropout_rate = dropout_rate)
            
        with tf1.variable_scope('Dense_Block_3'): 
            x = dense_block(x, nb_layers = 6, growth_rate = 16, isTraining = isTraining, dropout_rate = dropout_rate)       # 172 + 96 = 268 (16*16*16)
        with tf1.variable_scope('Transition_Block_3'):
            x, mid_x3 = transition_block(x, nb_filters = 276, isTraining = isTraining, dropout_rate = dropout_rate)
            
        with tf1.variable_scope('Dense_Block_4'): 
            x = dense_block(x, nb_layers = 6, growth_rate = 18, isTraining = isTraining, dropout_rate = dropout_rate)       # 268 + 108 = 376 (8*8*8)             
        with tf1.variable_scope('Cont_Block_1'):
            x = tf1.layers.conv3d_transpose(inputs = x, filters = 384, kernel_size = 2, strides = 2, padding = 'same', use_bias = False)
            x = tf.nn.relu(x)   
           # x = tf.concat(values = [x, mid_x3], axis = -1)                
            x = attention_block(x, mid_x3)  # 376+268
        with tf1.variable_scope('Conv_block_1'):
            x = tf1.layers.conv3d(inputs = x, filters = 276, kernel_size = 3, strides = 1, padding = 'same', use_bias =False)
            x = tf.nn.relu(x)
            x = tf1.layers.batch_normalization(inputs = x, training = True)
            
            x = tf1.layers.conv3d(inputs = x, filters = 276, kernel_size = 3, strides = 1, padding = 'same', use_bias = False)
            x = tf.nn.relu(x)
            x = tf1.layers.batch_normalization(inputs = x, training = True)

        with tf1.variable_scope('model_out_1'):
            out_1 = tf1.layers.conv3d_transpose(inputs = x, filters = 276, kernel_size = 4, strides = 4, padding = 'same', use_bias =False)
            out_1 = tf.nn.relu(out_1)
            out_1 = tf1.layers.conv3d(inputs = out_1, filters = nb_classes, kernel_size = 1, strides = 1, padding = 'same', use_bias  = False)
            
        with tf1.variable_scope('Atention_Block_2'):
            x = tf1.layers.conv3d_transpose(inputs = x, filters = 276, kernel_size = 2, strides = 2, padding = 'same', use_bias = False)
            x = tf.nn.relu(x)  
            # x = tf.concat(values = [x, mid_x2], axis = -1)                   
            x = attention_block(x, mid_x2) # 268+172
        with tf1.variable_scope('Conv_block_2'):
            x = tf1.layers.conv3d(inputs = x, filters = 180, kernel_size = 3, strides = 1, padding = 'same', use_bias = False)
            x = tf.nn.relu(x)
            x = tf1.layers.batch_normalization(inputs = x, training = True)
            x = tf1.layers.conv3d(inputs = x, filters = 180, kernel_size = 3, strides = 1, padding = 'same', use_bias = False)
            x = tf.nn.relu(x)
            x = tf1.layers.batch_normalization(inputs = x, training = True)

        with tf1.variable_scope('model_out_2'):
            out_2 = tf1.layers.conv3d_transpose(inputs = x, filters = 180, kernel_size = 2, strides = 2, padding = 'same', use_bias = False)
            out_2 = tf.nn.relu(out_2)
            out_2 = tf1.layers.conv3d(inputs = out_2, filters = nb_classes, kernel_size = 1, strides = 1, padding = 'same', use_bias  = False)
            
        with tf1.variable_scope('Attention_Block_3'):
            x = tf1.layers.conv3d_transpose(inputs = x, filters = 180, kernel_size = 2, strides = 2, padding = 'same', use_bias = False)
            x = tf.nn.relu(x)  
            # x = tf.concat(values = [x, mid_x1], axis = -1)                      
            x = attention_block(x, mid_x1) # 172+88
        with tf1.variable_scope('Conv_block_3'):
            x = tf1.layers.conv3d(inputs = x, filters = 96, kernel_size = 3, strides = 1, padding = 'same', use_bias = False)
            x = tf.nn.relu(x)
            x = tf1.layers.batch_normalization(inputs = x, training = True)
            x = tf1.layers.conv3d(inputs = x, filters = 96, kernel_size = 3, strides = 1, padding = 'same', use_bias = False)
            x = tf.nn.relu(x)
            x = tf1.layers.batch_normalization(inputs = x, training = True)

        with tf1.variable_scope('model_out_3'):
            out_3 = tf1.layers.conv3d(inputs = x, filters = nb_classes, kernel_size = 1, strides = 1, padding = 'same', use_bias  = False)
        
        return out_1, out_2, out_3


def Dense_Net(x, y, isTraining, dropout_rate, reuse = False):
    print('creating Dense_Net Discriminator ...')
    with tf1.variable_scope('Dense_Net') as scope:
        if reuse:
            print('reusing Dense_Net Discriminator ...')
            scope.reuse_variables()
        with tf1.variable_scope('focus_weight') as scope:
            gamma = tf.Variable(0.0,  name = "gamma")
        with tf1.variable_scope('Input_layer') as scope:
            back_x = tf.reshape(x[:,:,:,:,0], shape=[x.get_shape()[0], x.get_shape()[1],x.get_shape()[2], x.get_shape()[3], 1])
            fore_x = tf.reshape(x[:,:,:,:,1], shape=[x.get_shape()[0], x.get_shape()[1],x.get_shape()[2], x.get_shape()[3], 1]) 
            back_condition = tf.reshape(y[:,:,:,:,0], shape=[y.get_shape()[0], y.get_shape()[1],y.get_shape()[2], y.get_shape()[3], 1])
            fore_condition = tf.reshape(y[:,:,:,:,1], shape=[y.get_shape()[0], y.get_shape()[1],y.get_shape()[2], y.get_shape()[3], 1]) 
            fore = tf.concat(values=[fore_x, fore_condition], axis=-1)
            back = tf.math.multiply(tf.concat(values=[back_x, back_condition], axis=-1),gamma)
            x = tf.concat(values=[fore, back], axis=-1)
            x = tf1.layers.conv3d(inputs = x, filters = 16, kernel_size = 3, strides = 1, padding = 'same', use_bias = False)
        with tf1.variable_scope('Dense_block_1') as scope:
            x = dense_block(x, nb_layers= 4, growth_rate=8, isTraining=isTraining, dropout_rate=dropout_rate)
        with tf1.variable_scope('Transition_block_1') as scope:
            x, mid1 = transition_block(x, nb_filters= 48, isTraining=isTraining, dropout_rate=dropout_rate)
        with tf1.variable_scope('Dense_block_2') as scope:
            x = dense_block(x, nb_layers= 4, growth_rate=12, isTraining=isTraining, dropout_rate=dropout_rate)
        with tf1.variable_scope('Transition_block_2') as scope:
            x = tf1.layers.conv3d(inputs=x, filters= 96, kernel_size=1, strides=1, padding='same', use_bias=False)
            # ReLU
            x = tf.nn.relu(x)
            # BN
            x = tf1.layers.batch_normalization(inputs=x, training=True)
            if isTraining:
                x = tf1.layers.dropout(inputs=x, rate=dropout_rate)
            mid2 = tf1.layers.conv3d_transpose(inputs = x, filters = 48, kernel_size = 2, strides = 2, padding = 'same', use_bias = False)
            out = mid1+mid2
        return out
