#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 10:25:31 2018

@author: hans
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import argparse

NET = 'squeezenet' # squeezenet or mobilenet or mobilenetv2 or resnet ...
RESNET_LAYER_NUM = 101 # 50 or 101 or 152 or 200 or ...
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 27013 # number of training data
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 4996 # number of validation data 
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 4996 # number of testing data
NUM_LABELS = 1001 # number of classes
ORIGIN_IMAGE_SHAPE = 227 # origin image shape that equal to the size in list2bin_list.py
mean = [82.4088, 114.4588, 139.1493] # BGR mean values of training dataset
MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 10 # decrease learning rate every NUM_EPOCHS_PER_DECAY epochs
LEARNING_RATE_DECAY_FACTOR = 0.5 
WEIGHT_DECAY = 2e-4 # used for squeezenet and resnet

parser = argparse.ArgumentParser()

INITIAL_LEARNING_RATE = 0.01
STEPS_TO_VAL = 500
DEBUG = False
DATASET_DIR = 'data/' # Path to data directory.
MODEL_DIR = None # Directory where to write event logs and checkpoint.
FINETUNE_DIR = None
BATCH_SIZE = 64
LOG_FREQUENCY = 10 # How often to log results to the console.
MAX_STEPS = 100000 # Number of batches to run. If distributiong, all GPU batches.
LOG_DEVICE_PLACEMENT = False # Whether to log device placement.
USE_FP16 = False # Train the model using fp16.
MODE = 'training'
ISSYNC = False # only for distribution




parser.add_argument('--mode', type=str,default=MODE, help='Either `training` or `testing`.')
parser.add_argument('--lr', type=float, default=INITIAL_LEARNING_RATE)
parser.add_argument('--steps_to_val', type=int, default=STEPS_TO_VAL)
parser.add_argument('--debug', type=bool, default=DEBUG)
parser.add_argument('--dataset_dir', type=str, default=DATASET_DIR)
parser.add_argument('--model_dir', type=str, default=MODEL_DIR)
parser.add_argument('--finetune', type=str, default=FINETUNE_DIR)
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
parser.add_argument('--log_frequency', type=int, default=LOG_FREQUENCY)
parser.add_argument('--max_steps', type=int, default=MAX_STEPS)
parser.add_argument('--log_device_placement', type=bool, default=LOG_DEVICE_PLACEMENT)
parser.add_argument('--use_fp16', type=bool, default=USE_FP16)
parser.add_argument('--subset', type=str,default='train', help='Either `train` or `validation`.')
parser.add_argument('--num_preprocess_threads', type=int, default=16, help='Please make this a multiple of 4.')
parser.add_argument('--num_readers', type=int, default=16, help='Number of parallel readers during train.')
parser.add_argument('--image_size', type=int, default=299, help='Provide square images of this size.')
parser.add_argument('--input_queue_memory_factor', type=int, default=16, help='Size of the queue of preprocessed images, default is ideal but try smaller values, e.g, 4, 2 or 1, if host memory is constrained.')


# For distributed
PS_HOSTS = '10.156.129.78:2222' # Comma-separated list of hostname:port pairs
WORKER_HOSTS = '10.156.129.74:2223,10.156.129.86:2224' # Comma-separated list of hostname:port pairs

parser.add_argument('--issync', type=bool, default=ISSYNC)
parser.add_argument("--job_name", type=str,
                    help="One of 'ps', 'worker'")
parser.add_argument("--task_index", type=int,
                    help="Index of task within the job")

# No need to modify
if NET == 'squeezenet':
    IMAGE_RESIZE_SHAPE = 227 # image shape which suits for network
elif NET == 'mobilenet' or NET == 'mobilenetv2' or NET == 'resnet':
    IMAGE_RESIZE_SHAPE = 224
with tf.name_scope("mean_values"):
    b = tf.Variable(mean[0], dtype=tf.float32, name='b', trainable=False)
    g = tf.Variable(mean[1], dtype=tf.float32, name='g', trainable=False)
    r = tf.Variable(mean[2], dtype=tf.float32, name='r', trainable=False)
    MEAN = [b, g, r]
