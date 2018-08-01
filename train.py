#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 10:25:31 2018

@author: hans
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
from datetime import datetime

import tensorflow as tf
from tensorflow.python import debug as tfdbg
import arg_parsing
import image_processing
from net import squeezenet
from net import mobilenet
from net import mobilenetv2
from net import resnet

FLAGS = arg_parsing.parser.parse_args()


def _logits(images):
    if arg_parsing.NET == 'squeezenet':
        logits = squeezenet.inference(images)
    elif arg_parsing.NET == 'mobilenet':
        logits = mobilenet.inference(images)
    elif arg_parsing.NET == 'mobilenetv2':
        logits = mobilenetv2.inference(images)
    elif arg_parsing.NET == 'resnet':
        if arg_parsing.RESNET_LAYER_NUM==50:
            logits = resnet.resnet_v2_50(images)
        elif arg_parsing.RESNET_LAYER_NUM==101:
            logits = resnet.resnet_v2_101(images)
        elif arg_parsing.RESNET_LAYER_NUM==152:
            logits = resnet.resnet_v2_152(images)
        elif arg_parsing.RESNET_LAYER_NUM==200:
            logits = resnet.resnet_v2_200(images)
    return logits
    
def _loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _lr(global_step):
    with tf.name_scope("lr"):
        num_batches_per_epoch = arg_parsing.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
        decay_steps = int(num_batches_per_epoch * arg_parsing.NUM_EPOCHS_PER_DECAY)
        lr = tf.train.exponential_decay(FLAGS.lr,
                                        global_step,
                                        decay_steps,
                                        arg_parsing.LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)
        tf.summary.scalar('lr', lr)
    return lr

def _optimization(total_loss, global_step,lr, issync, num_workers=0):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
    if issync:
        global sync_replicas_hook
        opt = tf.train.SyncReplicasOptimizer(opt,
                                             replicas_to_aggregate=num_workers,
                                             total_num_replicas=num_workers,
                                             use_locking=True)
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        sync_replicas_hook = opt.make_session_run_hook(is_chief=(FLAGS.task_index == 0))
    else:
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    
    variable_averages = tf.train.ExponentialMovingAverage(
                        arg_parsing.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
    return train_op

def val(train_loss,dataset):
    with tf.name_scope("val_process"):
        with tf.device('/cpu:0'):
            val_images, val_labels = image_processing.distorted_inputs(
                dataset,
                num_preprocess_threads=FLAGS.num_preprocess_threads)
        val_logits = _logits(val_images)
        val_loss = _loss(val_logits, val_labels)

        val_acc = tf.nn.in_top_k(val_logits,val_labels,1)
        val_acc_sum = tf.cast(val_acc,tf.float32)
        val_acc_sum = tf.reduce_mean(val_acc_sum)

    with tf.name_scope("loss"):
        tf.summary.scalar('train_loss', train_loss)
        tf.summary.scalar('val_loss', val_loss)
    return val_acc_sum

def train(dataset):
    print('START')
    if FLAGS.issync:
        raise ValueError("Please set 'issync' to False when non-distribution")
    global_step = tf.Variable(0, dtype=tf.int32, name='global_step', trainable=False)
    lr=_lr(global_step)
    with tf.name_scope("train_process"):
        with tf.device('/cpu:0'):
            images, labels = image_processing.distorted_inputs(
                dataset,
                num_preprocess_threads=FLAGS.num_preprocess_threads
            )
        logits = _logits(images)
        loss = _loss(logits, labels)  
        train_op = _optimization(loss, global_step,lr, FLAGS.issync)
#    with tf.name_scope("global_step"):
#        tf.summary.scalar('global_step', global_step)
    val_step = int(math.ceil(arg_parsing.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL/FLAGS.batch_size))
    val_acc_sum = val(loss,dataset)
    all_hooks=[tf.train.NanTensorHook(loss)]
    if FLAGS.debug:
        all_hooks.append(tfdbg.LocalCLIDebugHook(ui_type='curses'))
    if FLAGS.finetune:
        print('Finetune from %s'%FLAGS.finetune)
        saver = tf.train.Saver()
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth=True
    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.model_dir,
            hooks=all_hooks,
            config=config,
            save_summaries_steps=100,
            save_summaries_secs=None,
            log_step_count_steps=None) as sess:
        if FLAGS.finetune:
            print('Load Pre-trained model...')
            ckpt = tf.train.get_checkpoint_state(FLAGS.finetune)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise ValueError('Failed to load model.')
            print('-------------------------')
        total_loss = 0
        start_time = time.time()
        for i in range(1, FLAGS.max_steps+1):
            _,loss_value = sess.run([train_op,loss])
            total_loss += loss_value
            if i % FLAGS.log_frequency == 0:
                current_time = time.time()
                duration = current_time - start_time
                eg_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                sec_per_batch = float(duration / FLAGS.log_frequency)
                avg_loss = total_loss/i
                print('%s: training step %d cur loss = %.4f avg loss = %.4f (%.1f images/sec %.3f sec/batch)'
                      % (datetime.now(), i, loss_value, avg_loss, eg_per_sec, sec_per_batch))
                start_time = time.time()
            if i % FLAGS.steps_to_val == 0:
                total_val_accu=0
                for j in range(val_step):
                    total_val_accu+=sess.run(val_acc_sum)
                print('%s: validation total accuracy = %.4f (%.3f sec %d batches)'
                      % (datetime.now(), total_val_accu/float(val_step), float(time.time()-start_time), val_step))
                start_time = time.time()
                
def train_dis_(dataset):
    ps_hosts = arg_parsing.PS_HOSTS.split(",")
    worker_hosts = arg_parsing.WORKER_HOSTS.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
        server.join()
    if FLAGS.job_name == "worker":
        print('START')
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" 
                                                      %FLAGS.task_index,cluster=cluster)):
            global_step = tf.Variable(0, dtype=tf.int32, name='global_step', trainable=False)
            
            lr=_lr(global_step)
            with tf.name_scope("train_process"):
                with tf.device('/cpu:0'):
                    images, labels = image_processing.distorted_inputs(
                        dataset,
                        num_preprocess_threads=FLAGS.num_preprocess_threads
                    )
                logits = _logits(images)
                loss = _loss(logits, labels)
                train_op = _optimization(loss, global_step,lr, FLAGS.issync, len(worker_hosts))
#            with tf.name_scope("global_step"):
#                tf.summary.scalar('global_step', global_step)

            val_acc_sum = val(loss,dataset)
            class _LoggerHook(tf.train.SessionRunHook):
                def begin(self):
                    self._local_step = 0
                    self._start_time = time.time()
                    self._total_loss = 0
                def before_run(self, run_context):
                    self._local_step += 1
                    return tf.train.SessionRunArgs(loss)
                def after_run(self, run_context, run_values):
                    self._step = run_context.session.run(global_step)
                    loss_value = run_values.results

                    self._total_loss += loss_value
                    if self._step % FLAGS.log_frequency == 0:
                        current_time = time.time()
                        duration = current_time - self._start_time
                        self._start_time = current_time
                        avg_loss = self._total_loss/self._local_step
                        eg_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                        sec_per_batch = float(duration / FLAGS.log_frequency)
                        print('%s: training step %d cur loss = %.4f avg loss = %.4f (%.1f images/sec %.3f sec/batch)'
                              % (datetime.now(), self._step, loss_value, avg_loss, eg_per_sec, sec_per_batch))
            class _ValHook(tf.train.SessionRunHook):
                def begin(self):
#                    self._step = 0
                    self._val_step = int(math.ceil(arg_parsing.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL/FLAGS.batch_size))
                def before_run(self, run_context):
#                    self._step += 1
                    self._total_val_accu = 0
                    self._start_time = time.time()
                def after_run(self, run_context, run_values):
#                    if FLAGS.issync:
                    self._step = run_context.session.run(global_step)
                    if self._step % FLAGS.steps_to_val == 0:
                        if (FLAGS.task_index==0 and FLAGS.issync) or not FLAGS.issync:
                            for j in range(self._val_step):
                                self._total_val_accu+=run_context.session.run(val_acc_sum)
                            print('%s: step %d validation accuracy = %.4f (%.3f sec %d batches)'
                                  % (datetime.now(), self._step, self._total_val_accu/float(self._val_step), float(time.time()-self._start_time), self._val_step))

            class _ExitHook(tf.train.SessionRunHook): # same as StopAtStepHook
                def begin(self):
                    self._val_step = int(math.ceil(arg_parsing.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL/FLAGS.batch_size))
                def before_run(self, run_context):
                    self._total_val_accu = 0
                    self._start_time = time.time()
                def after_run(self, run_context, run_values):
                    self._step = run_context.session.run(global_step)
                    if self._step>=FLAGS.max_steps:
                        if FLAGS.task_index==0 and not FLAGS.issync:
                            for j in range(self._val_step*2):
                                self._total_val_accu+=run_context.session.run(val_acc_sum)
                            print('%s: last step %d validation final accuracy = %.4f (%.3f sec(2 times) %d batches)'
                                  % (datetime.now(), self._step, self._total_val_accu/float(self._val_step*2), float(time.time()-self._start_time), self._val_step))
                        run_context.request_stop()

#            all_hooks=[tf.train.NanTensorHook(loss), tf.train.StopAtStepHook(last_step=FLAGS.max_steps), _LoggerHook(), _ValHook()]
            all_hooks=[tf.train.NanTensorHook(loss), _LoggerHook(), _ValHook(), _ExitHook()]
            if FLAGS.issync:
                all_hooks.append(sync_replicas_hook)
            if FLAGS.debug:
                all_hooks.append(tfdbg.LocalCLIDebugHook(ui_type='curses'))
            if FLAGS.finetune:
                print('Finetune from %s'%FLAGS.finetune)
                saver = tf.train.Saver()
            config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
            config.gpu_options.allow_growth=True
            with tf.train.MonitoredTrainingSession(
                    master=server.target,
                    is_chief=(FLAGS.task_index == 0),
                    checkpoint_dir=FLAGS.model_dir,
                    hooks=all_hooks,
                    config=config,
                    save_summaries_steps=100,
                    save_summaries_secs=None,
                    log_step_count_steps=None) as sess:
                if FLAGS.finetune:
                    print('Load Pretrained model')
                    ckpt = tf.train.get_checkpoint_state(FLAGS.finetune)
                    if ckpt and ckpt.model_checkpoint_path:
                        saver.restore(sess, ckpt.model_checkpoint_path)
                    print('-------------------------')
                while not sess.should_stop():
                    sess.run(train_op)
                print('DONE')
