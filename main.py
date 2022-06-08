# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 11:25:33 2019

@author: chenying
"""
import os
import glob
from os.path import join
import SimpleITK as sitk
import time

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from tensorflow.keras.utils import to_categorical

from utils import readimage, standardize_info, ref_info_for_write, stitch_tiles, volume_to_tiles, get_tiles_count, write_seg_result
from model import A_Dense_UNet
from model import Dense_Net
from loss import conf_weighted_crossentropy_with_ou
from metric import metric_iou_dice

"""
Data paths 
"""
volumes_dir = './data/train/raw/'
volumes_gt_dir = './data/train/gt/'

val_volumes_dir = './data/val/raw/'
val_volumes_gt_dir = './data/val/gt/'

test_volumes_dir = './data/test/raw/'
test_volumes_gt_dir = './data/test/gt/'

netstore_path = './netstore/'
pred_result_dir = './data/infer/'
training_volumes = glob.glob(join(volumes_dir,'*-Cr-MRA.nii.gz'))
validation_volumes = glob.glob(join(val_volumes_dir,'*-Cr-MRA.nii.gz')) 
testing_volumes = glob.glob(join(test_volumes_dir,'*-Cr-MRA.nii.gz')) 
"""
Settings
"""
tile_size = [64, 64, 64]
overlap_size = [36, 4, 4]
batch_size = 2
tile_channel = 1
nb_classes = 2
dropout_rate = 0.2
isTraining = True  # False when testing
restore_epochs = None  
tf1.disable_eager_execution()
use_GAN = True # False when only training segmentor
steps_k = 2

"""
Model
"""
G_input = tf1.placeholder(dtype=tf.float32, shape=[batch_size, tile_size[0], tile_size[1], tile_size[2], tile_channel])
G_supervise = tf1.placeholder(dtype=tf.float32, shape=[batch_size, tile_size[0], tile_size[1], tile_size[2], nb_classes])
G_output_1, G_output_2, G_output_3= A_Dense_UNet(G_input, nb_classes = nb_classes, isTraining = isTraining, dropout_rate = dropout_rate, reuse = False)

G_sample_1 = tf.nn.softmax(G_output_1)
G_sample_2 = tf.nn.softmax(G_output_2)
G_sample_3 = tf.nn.softmax(G_output_3)
## Discriminator
if use_GAN:
    d_input_fake = tf.math.multiply(G_sample_3, G_input)
    d_input_real = tf.math.multiply(G_supervise, G_input)
    d_condition = G_supervise
    
    d_output_fake= Dense_Net(d_input_fake, d_condition, isTraining=isTraining, dropout_rate= dropout_rate, reuse=False)
    d_output_real= Dense_Net(d_input_real, d_condition, isTraining=isTraining, dropout_rate= dropout_rate, reuse=True)
    
"""
Loss function
"""
# loss of Generator
G_loss_CEA = conf_weighted_crossentropy_with_ou(outputs=G_output_1, targets = G_supervise, classes = nb_classes) + conf_weighted_crossentropy_with_ou(outputs=G_output_2, targets = G_supervise, classes = nb_classes) + conf_weighted_crossentropy_with_ou(outputs=G_output_3, targets = G_supervise, classes = nb_classes)

if use_GAN:  
    G_loss_GAN = tf1.reduce_mean(tf1.scalar_mul(-1,d_output_fake))
    G_loss = G_loss_CEA + G_loss_GAN
else:
    G_loss = G_loss_CEA
    
# loss of Discriminator
if use_GAN: 
    D_loss_real = tf1.reduce_mean(tf1.scalar_mul(-1,d_output_real))
    D_loss_fake = tf1.reduce_mean(d_output_fake)
    D_loss =  D_loss_real + D_loss_fake

"""
Optimizer & Trainer
"""
t_vars = tf1.trainable_variables()
lr_G_start = 0.0
global_steps_g = 0
global_steps_d = 0
if restore_epochs is not None:
    global_steps_g = restore_epochs * 578 * len(training_volumes)
    global_steps_d = steps_k * restore_epochs * 578 * len(training_volumes)
## Optimizer & Trainer of Generator
lr_G = tf1.train.exponential_decay(learning_rate = lr_G_start, global_step = global_steps_g, decay_steps = 5 * 578 * len(training_volumes), decay_rate = 0.9)
g_parameters = [var for var in t_vars if var.name.startswith('A_Dense_UNet')]
optim_G = tf1.train.GradientDescentOptimizer(learning_rate = lr_G)
trainer_G = optim_G.minimize(G_loss, var_list = g_parameters)
## Optimizer & Trainer of Discriminator
lr_D_start = 0.001
lr_D = tf1.train.exponential_decay(learning_rate = lr_D_start, global_step = global_steps_d, decay_steps = 5 * 578 * len(training_volumes), decay_rate = 0.9)
d_parameters = [var for var in t_vars if var.name.startswith('Dense_Net')]
optim_D = tf1.train.GradientDescentOptimizer(learning_rate = lr_D)
trainer_D = optim_D.minimize(D_loss, var_list = d_parameters)
gamma_parameters = [var for var in d_parameters if 'focus_weight' in var.name]

clip_ops = []
for var in gamma_parameters:
    clip_ops.append(
        tf.assign(var, tf.clip_by_value(var, 0.0, 1.0)))
"""
training & testing
"""
model_folder = 'FSegAN_CV1_K4_'  + str(lr_G_start) + '_' + str(lr_D_start) + '_' + str(len(training_volumes))
def create_model_name(epoch_number):
    return netstore_path + model_folder + '/epoch_'+ str(epoch_number) +'.model'
saver = tf1.train.Saver(max_to_keep=60)

config = tf1.ConfigProto()
config.gpu_options.allow_growth = True

with tf1.Session(config=config) as sess:  
    if restore_epochs is not None:
        fname = create_model_name(restore_epochs)
        if tf1.train.checkpoint_exists(fname):
            print('restoring model weights from:', fname)
            saver.restore(sess, fname)  
    else:
        tf1.global_variables_initializer().run()
        
    if isTraining:
        print('START TRAINING...')           
        current_epoch = 1
        if restore_epochs is not None:
            current_epoch = restore_epochs + 1
        print('Epoch:', current_epoch)
        for index in range(len(training_volumes)):
            print('Training Volume:', training_volumes[index])
            training_volume =training_volumes[index]
            training_gt = training_volume.replacce(volumes_dir, volumes_gt_dir).replace('MRA.nii', 'MRA_GT.nii')
            volume_raw = readimage(os.path.abspath(training_volume))
            tiles_raw = volume_to_tiles(volume_raw, tile_size, overlap_size, GT = False)
            volume_gt = readimage(os.path.abspath(training_gt))
            tiles_gt = volume_to_tiles(volume_gt, tile_size, overlap_size, GT = True)
            if use_GAN:
                tiles_raw_d = np.tile(tiles_raw, (steps_k, 1,1,1))                   
                tiles_gt_d =  np.tile(tiles_raw, (steps_k, 1,1,1))
                print('training GAN:')
                for start_index in range(0, tiles_raw.shape[0], batch_size):
                    for k in range(steps_k):
                        x_train_d = tiles_raw_d[(steps_k * start_index + k * batch_size) : (steps_k * start_index+ (k+1) * batch_size), :,:,:,np.newaxis]
                        y_train_d = np.array(tiles_gt_d[(steps_k * start_index + k * batch_size) : (steps_k * start_index+ (k+1) * batch_size), :, :,:, np.newaxis]>0).astype(int)
                        y_train_d = to_categorical(y_train_d, nb_classes)   
                        update_ops = tf1.get_collection(tf1.GraphKeys.UPDATE_OPS)
                        with tf.control_dependencies(update_ops):                            
                            _, D_loss_value, D_loss_real_value, D_loss_fake_value= sess.run([trainer_D, D_loss, D_loss_real, D_loss_fake], feed_dict = {G_input:x_train_d, G_supervise:y_train_d})                            
                            print('D_loss:{:.8f}'.format(D_loss_value), 'D_loss_real:{:.8f}'.format(D_loss_real_value),'D_loss_fake:{:.8f}'.format(D_loss_fake_value))
                            _ = sess.run(clip_ops)
                            global_steps_d = global_steps_d + 1
                    x_train_g = tiles_raw[start_index:start_index+batch_size,:,:,:,np.newaxis]
                    y_train_g = np.array(tiles_gt[start_index:start_index+batch_size,:,:,:,np.newaxis]>0).astype(int)
                    y_train_g = to_categorical(y_train_g, nb_classes)
                    update_ops = tf1.get_collection(tf1.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_ops): 
                        _, G_loss_value, G_loss_CEA_value, G_loss_GAN_value= sess.run([trainer_G, G_loss, G_loss_CEA, G_loss_GAN], feed_dict = {G_input:x_train_g, G_supervise:y_train_g})
                        print('G_loss:{:.8f}'.format(G_loss_value),'G_loss_CEA:{:.8f}'.format(G_loss_CEA_value), 'G_loss_GAN:{:.8f}'.format(G_loss_GAN_value))
                        global_steps_g = global_steps_g + 1                           
            else:
                print('training generator only:')
                for start_index in range(0, tiles_raw.shape[0], batch_size):                       
                    x_train_g =  tiles_raw[start_index:start_index+batch_size,:,:,:,np.newaxis]
                    y_train_g = np.array(tiles_gt[start_index:start_index+batch_size,:,:,:,np.newaxis]>0).astype(int)
                    y_train_g = to_categorical(y_train_g, nb_classes)
                    update_ops = tf1.get_collection(tf1.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_ops): 
                        _, G_loss_value, G_loss_CEA_value, G_loss_GAN_value= sess.run([trainer_G, G_loss, G_loss_CEA, G_loss_GAN], feed_dict = {G_input:x_train_g, G_supervise:y_train_g})
                        print('G_loss:{:.8f}'.format(G_loss_value),'G_loss_CEA:{:.8f}'.format(G_loss_CEA_value), 'G_loss_GAN:{:.8f}'.format(G_loss_GAN_value))
                        global_steps_g = global_steps_g + 1                         
            saver.save(sess,create_model_name(current_epoch))
            
            valid_loss_G, valid_loss_CEA_G = [], []
            for index in range(len(validation_volumes)):
                print('START Validation...')
                print('Validation Volume:', validation_volumes[index])
                validation_volume = validation_volumes[index]
                validation_gt = validation_volume.replace(val_volumes_dir, val_volumes_gt_dir).replace('MRA.nii', 'MRA_GT.nii')
                volume_raw = readimage(os.path.abspath(validation_volume))
                tiles_raw = volume_to_tiles(volume_raw, tile_size, overlap_size,  GT = False)
                volume_gt = readimage(os.path.abspath(validation_gt))
                tiles_gt = volume_to_tiles(volume_gt, tile_size, overlap_size,  GT = True)
                #tiles_gt_dist = distance_transform(tiles_gt)
                for tile in range(0,tiles_raw.shape[0],batch_size):
                    x_valid = tiles_raw[tile:tile+batch_size,:,:,:,np.newaxis]
                    y_valid = np.array(tiles_gt[tile:tile+batch_size,:,:,:,np.newaxis]>0).astype(int)
                    y_valid = to_categorical(y_valid, nb_classes)
                    G_loss_value, G_loss_CEA_value = sess.run([G_loss,G_loss_CEA], feed_dict = {G_input:x_valid, G_supervise:y_valid})
                    valid_loss_G.append(G_loss_value)
                    valid_loss_CEA_G.append(G_loss_CEA_value)
            f=open('validationloss.txt', 'a')
            f.write('Epoch' + str(current_epoch) + ':'+'\n' )
            f.write(str(np.mean(valid_loss_G)) + '\n')
            f.write(str(np.mean(valid_loss_CEA_G)) + '\n')
            f.close()
            current_epoch = current_epoch+1  

    else:
        print('Start Testing...')
        fore_iou, fore_dice = [], []
        for index in range(len(testing_volumes)):
            print('Testing Volume:', testing_volumes[index])
            
            testing_volume = testing_volumes[index]
            volume_size, ref_origin, ref_spacing, ref_direction = ref_info_for_write(testing_volume)
            testing_gt = testing_volume.replace(test_volumes_dir, test_volumes_gt_dir).replace('MRA.nii', 'MRA_GT.nii')
            volume_raw = readimage(os.path.abspath(testing_volume))
            tiles_raw = volume_to_tiles(volume_raw, tile_size, overlap_size,  GT = False)
            
            volume_gt = readimage(os.path.abspath(testing_gt))
            tiles_gt = volume_to_tiles(volume_gt, tile_size, overlap_size, GT = True)
            
            tiles_seg_result = np.zeros(shape=[tiles_raw.shape[0], tile_size[0], tile_size[1], tile_size[2], nb_classes])
            tiles_seg_gt = np.zeros(shape = [tiles_raw.shape[0], tile_size[0], tile_size[1], tile_size[2], nb_classes])
            
            for tile in range(0,tiles_raw.shape[0],batch_size):
                x_test = tiles_raw[tile:tile+batch_size,:,:,:,np.newaxis]
                y_test = tiles_gt[tile:tile+batch_size,:,:,:,np.newaxis]

                y_test = to_categorical(y_test, nb_classes)
                tiles_seg_gt[tile:tile+batch_size, :,:,:,:] = y_test

                result = sess.run(G_sample_3,feed_dict = {G_input:x_test}) 
                tiles_seg_result[tile:tile + batch_size, :, :, :, 1] = np.argmax(result, axis=-1)
                tiles_seg_result[tile:tile + batch_size, :, :, :, 0] = 1 - np.argmax(result, axis=-1)
            fore_viou, fore_vdice =  metric_iou_dice(tiles_seg_result[:,:,:,:,1], tiles_seg_gt[:,:,:,:,1])
            print('volume_fore_iou_' + str(index), fore_viou)
            print('volume_fore_dice_' + str(index), fore_vdice)
            fore_iou.append(fore_viou)
            fore_dice.append(fore_vdice)
            volume_seg_result = stitch_tiles(tiles_seg_result[:,:,:,:,1], volume_size, tile_size, overlap_size)
            write_seg_result(volume_seg_result, ref_origin, ref_spacing, ref_direction, testing_volumes[index].replace(test_volumes_dir, pred_result_dir).replace('.nii.gz', '_Pred.nii.gz'))
        print('fore_ave_iou:', np.mean(fore_iou))
        print('fore_ave_dice:', np.mean(fore_dice))

    
       

