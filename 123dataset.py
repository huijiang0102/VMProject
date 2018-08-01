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
FLAGS = arg_parsing.parser.parse_args()

def _generate_image_and_label_batch(image, label, min_q_eg, batch_size, shuffle):
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
                              [image, label],
                              batch_size=batch_size,
                              num_threads=num_preprocess_threads,
                              capacity=min_q_eg + 3 * batch_size,
                              min_after_dequeue=min_q_eg)
    else:
        images, label_batch = tf.train.batch(
                              [image, label],
                              batch_size=batch_size,
                              num_threads=num_preprocess_threads,
                              capacity=min_q_eg + 3 * batch_size)

    return images, tf.reshape(label_batch, [batch_size])

def read_and_decode(data_files):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer(data_files,shuffle=True,capacity=16)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [arg_parsing.ORIGIN_IMAGE_SHAPE, arg_parsing.ORIGIN_IMAGE_SHAPE, 3])
    img = tf.image.resize_images(img, [arg_parsing.IMAGE_RESIZE_SHAPE,arg_parsing.IMAGE_RESIZE_SHAPE])
    red, green, blue = tf.split(img, 3, 2)
    blue = tf.cast(blue, tf.float32)-arg_parsing.MEAN[0]
    green = tf.cast(green, tf.float32)-arg_parsing.MEAN[1]
    red = tf.cast(red, tf.float32)-arg_parsing.MEAN[2]
    img = tf.concat([blue, green, red], 2)
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    return img, label

def process_inputs(mode):
    data_dir = os.path.join(FLAGS.dataset_dir)
    # tf_record_pattern = os.path.join(FLAGS.data_dir, '%s-*' % self.subset)
    # if not data_files:
    #     print('No files found for dataset %s/%s at %s' % (self.name,
    #                                                       self.subset,
    #                                                       FLAGS.data_dir))
    #
    #     self.download_message()
    #     exit(-1)
    # return data_files

    if mode == "training":
        tf_record_pattern = os.path.join(data_dir, 'train-*')
    elif mode == "testing":
        tf_record_pattern = os.path.join(data_dir, 'val.tfrecords-*')
    elif mode == "val":
        tf_record_pattern = os.path.join(data_dir, 'validation-*')
    data_files = tf.gfile.Glob(tf_record_pattern)
    if not data_files:
        raise ValueError('Failed to find file: ' + data_files)
    image, label = read_and_decode(data_files)

    min_queue_examples = int(0.4*arg_parsing.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)

    shuffle = True if mode == "training" else False
    images, labels = _generate_image_and_label_batch(image, label,
                                                     min_queue_examples, arg_parsing.BATCH_SIZE,
                                                     shuffle=shuffle)


    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)

    return images, labels

#images, labels = process_inputs("training")
#image, label = read_and_decode("../data/test.tfrecords")
