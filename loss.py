# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1


def conf_weighted_crossentropy_with_ou(outputs, targets, classes):
    xent = tf1.losses.softmax_cross_entropy(onehot_labels=targets, logits=outputs)
    y_true_bin = tf1.argmax(targets, axis=-1)
    
    y_pred = tf.nn.softmax(outputs)
    y_pred_bin = tf1.argmax(y_pred, axis=-1)
    
    epsilon = 1e-4
    C = 0
    for c in range(classes):
        c_true = tf.cast(tf.math.equal(y_true_bin, c), dtype=tf.float32)
        w = 1./(tf1.reduce_sum(c_true) + epsilon)
        
        conf = c_true *( 1 + tf.math.abs((targets[:,:,:,:,c]-y_pred[:,:,:,:,c])))
        C = C + tf1.reduce_sum(xent * conf * w)
        if c==1:
            y_pred_over = tf.cast(tf.math.not_equal(y_true_bin, c),tf.float32) * tf.cast(tf.math.equal(y_pred_bin, c), tf.float32)
            y_pred_under = tf.cast(tf.math.equal(y_true_bin, c),tf.float32) * tf.cast(tf.math.not_equal(y_pred_bin, c), tf.float32)
            #y_pred_error = tf.cast(tf.not_equal(y_true_bin, y_pred_bin), dtype = tf.float32)
            
            #w_o =  tf.reduce_sum(y_pred_over) / ((tf.reduce_sum(y_pred_error)+epsilon)* (tf.reduce_sum(y_pred_error)+epsilon))
            #w_u =  tf.reduce_sum(y_pred_under) / ((tf.reduce_sum(y_pred_error)+epsilon) * (tf.reduce_sum(y_pred_error)+epsilon))
            w_o = 1./(tf1.reduce_sum(y_pred_over)+epsilon)
            w_u = 1. / (tf1.reduce_sum(y_pred_under) + epsilon)
            C = C + tf1.reduce_sum(xent * y_pred_over * tf.math.abs((targets[:,:,:,:,c]-y_pred[:,:,:,:,c])) * w_o )
            C = C + tf1.reduce_sum(xent * y_pred_under * tf.math.abs((targets[:,:,:,:,c]-y_pred[:,:,:,:,c])) * w_u )
    return C