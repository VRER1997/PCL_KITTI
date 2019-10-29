# -*- coding: utf-8 -*-
# @Time    : 2019/10/23 上午10:48
# @Author  : Gao Xiaosa


import tensorflow as tf
import numpy as np


class ClassifyNet(object):
    def __init__(self, input, training=True):
        self.input = input
        self.training = training
        self.targets = tf.placeholder(tf.float32, [None, 2])

        # (5, 10, 10, 10, 32)
        temp_conv = ConvMD(3, 32, 16, 3, (2, 2, 2), (1, 1, 1), self.input)
        # (5, 5, 5, 5, 16)
        # temp_conv = ConvMD(3, 16, 16, 3, (1, 1, 1), (0, 1, 1), temp_conv)
        temp_conv = ConvMD(3, 16, 16, 3, (2, 1, 1), (0, 1, 1), temp_conv)
        temp_conv = tf.transpose(temp_conv, perm=[0, 2, 3, 4, 1])
        # (5, 5, 5, 16, 2)
        temp_conv = tf.reshape(temp_conv, shape=(-1, 5, 5, 32))

        # (5, 5, 5, 32)
        temp_conv = ConvMD(2, 32, 16, 3, (2, 2), (0, 0), temp_conv)
        # (5, 2, 2, 16)
        temp_conv = ConvMD(2, 16, 8, 3, (1, 1), (1, 1), temp_conv)

        # (5, 2, 2, 8)
        flatten = tf.layers.flatten(temp_conv)
        dense = tf.layers.Dense(8, activation=tf.nn.relu).apply(flatten)
        dense1 = tf.layers.Dense(2, activation=None).apply(dense)
        pred = tf.nn.softmax(dense1, name="pred")

        cross_entropy = -tf.reduce_sum(self.targets * tf.log(pred))
        self.cls_loss = cross_entropy

        acc_num = tf.equal(tf.argmax(pred, 1), tf.argmax(self.targets, 1))
        self.acc_rate = tf.reduce_mean(tf.cast(acc_num, tf.float32))
        self.prob_output = pred


def ConvMD(M, Cin, Cout, k, s, p, input, training=True, activate=None):
    temp = np.array(p)
    temp = np.lib.pad(temp, (1, 1), 'constant', constant_values=(0, 0))
    if M == 2:
        padding = (np.array(temp)).repeat(2).reshape((4, 2))
        pad = tf.pad(input, padding, "CONSTANT")
        #  ''', reuse=tf.AUTO_REUSE'''
        temp_conv = tf.layers.conv2d(pad, Cout, k, strides=s, padding='valid')
    if M == 3:
        padding = (np.array(temp)).repeat(2).reshape((5, 2))
        pad = tf.pad(input, padding, "CONSTANT")
        temp_conv = tf.layers.conv3d(pad, Cout, k, strides=s, padding='valid')

    temp_conv = tf.layers.batch_normalization(
        temp_conv, axis=-1, fused=True, training=training)
    if activate:
        return tf.nn.relu(temp_conv)
    else:
        return temp_conv