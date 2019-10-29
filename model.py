# -*- coding: utf-8 -*-
# @Time    : 2019/10/21 下午8:06
# @Author  : Gao Xiaosa

import tensorflow as tf

from group_pointcloud import FeatureNet
from classify import ClassifyNet


class MODEL(object):

    def __init__(self,
                 # cls='car',
                 batch_size=2,
                 learning_rate=0.001,
                 is_train=True
                 ):
        # self.cls = cls
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(
            float(learning_rate), trainable=False, dtype=tf.float32)
        self.epoch = tf.Variable(0, trainable=False)
        self.global_step = tf.Variable(0, trainable=False)
        self.epoch_add_op = self.epoch.assign(self.epoch+1)
        lr = tf.train.exponential_decay(
            self.learning_rate, self.global_step, 10000, 0.96)

        feature = FeatureNet(training=is_train, batch_size=self.batch_size)
        classify = ClassifyNet(training=is_train, input=feature.outputs)

        self.vox_feature = feature.feature
        self.vox_number = feature.number
        self.vox_coordinate = feature.coordinate

        self.targets = classify.targets
        self.cls_loss = classify.cls_loss
        self.acc_rate = classify.acc_rate
        self.prob_output = classify.prob_output

        self.opt = tf.train.AdamOptimizer(lr).minimize(self.cls_loss, global_step=self.global_step)
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2,
                                    max_to_keep=10, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def train_step(self, sess, data, train=False, summary=False):
        # (tag, vox_feature, vox_number, vox_coordinate)
        tag, vox_feature, vox_number, vox_coordinate = data
        input_feed = {}
        input_feed[self.vox_feature] = vox_feature
        input_feed[self.vox_number] = vox_number
        input_feed[self.vox_coordinate] = vox_coordinate
        input_feed[self.targets] = tag

        output_feed = [self.opt, self.cls_loss, self.acc_rate]
        return sess.run(output_feed, input_feed)

    def validate_step(self, sess,  data, summary=False):
        tag, vox_feature, vox_number, vox_coordinate = data
        input_feed = {}
        input_feed[self.vox_feature] = vox_feature
        input_feed[self.vox_number] = vox_number
        input_feed[self.vox_coordinate] = vox_coordinate
        input_feed[self.targets] = tag

        output_feed = [self.cls_loss, self.acc_rate]
        return sess.run(output_feed, input_feed)

    def predict_step(self, sess, data, summary=False):
        pass

