#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 10:25:31 2018

@author: hans
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

import arg_parsing
import train
from imagenet_data import ImagenetData

import test
FLAGS = arg_parsing.parser.parse_args()
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def printInfo():
    print('-------------------------')
    print('Network: %s' %arg_parsing.NET)
    if arg_parsing.NET == 'resnet':
        print('Layer nums: %d' %arg_parsing.RESNET_LAYER_NUM)
    print('Initial learning rate: %f' %FLAGS.lr)
    print('Steps to val: %d' %FLAGS.steps_to_val)
    print('debug: %s' %FLAGS.debug)
    print('Dataset dir: %s' %FLAGS.dataset_dir)
    print('Model dir: %s' %FLAGS.model_dir)
    if FLAGS.finetune:
        print('Finetune dir: %s' %FLAGS.finetune)
    print('Batch size: %d' %FLAGS.batch_size)
    print('Log frequency: %d' %FLAGS.log_frequency)
    print('Max steps: %d' %FLAGS.max_steps)
    print('Log device placement: %s' %FLAGS.log_device_placement)
    print('Use fp16: %s' %FLAGS.use_fp16)
    if FLAGS.job_name:
        print('\nDistuibution info: ')
        print('Issync: %s' %FLAGS.issync)
        if FLAGS.issync:
            print('PS HOSTS: %s' %arg_parsing.PS_HOSTS)
            print('WORKER HOSTS: %s' %arg_parsing.WORKER_HOSTS)
    print('-------------------------')

def main(argv=None):
    dataset = ImagenetData(subset=FLAGS.subset)
    if FLAGS.job_name:
        assert FLAGS.job_name in ['ps', 'worker'], 'job_name must be ps or worker'

#        if tf.gfile.Exists(arg_parsing.MODEL_DIR):
#            tf.gfile.DeleteRecursively(arg_parsing.MODEL_DIR)
#        else:
#           tf.gfile.MakeDirs(FLAGS.model_dir)
        printInfo()
        print("dataset",dataset)
        train.train_dis_(dataset)
    else:
        assert dataset.data_files()
        # if (FLAGS.mode == 'testing'):
        #     test.test(FLAGS.mode)
        # else:
        printInfo()
        print("dataset",dataset)
        train.train(dataset)
#    else:
#        raise ValueError("set --mode as 'training' or 'testing' or 'training_dis'")

if __name__ == '__main__':
    tf.app.run()
