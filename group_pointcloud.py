# -*- coding: utf-8 -*-
# @Time    : 2019/10/21 下午8:32
# @Author  : Gao Xiaosa

import os
import time
import numpy as np
import tensorflow as tf


class VFELayer(object):

    def __init__(self, out_channels):
        super(VFELayer, self).__init__()
        self.units = int(out_channels/2)
        self.dense = tf.layers.Dense(
            self.units, tf.nn.relu)
        self.batch_norm = tf.layers.BatchNormalization(fused=True)

    def apply(self, inputs, mask, training):
        # (K, 10, 7)
        pointwise = self.batch_norm.apply(self.dense.apply(inputs), training)
        # (K, 10, 4)
        aggregated = tf.reduce_max(pointwise, axis=1, keep_dims=True)
        # (K, 1, 4)
        repeated = tf.tile(aggregated, [1, 10, 1])
        # (K, 10, 4)
        concatenated = tf.concat([pointwise, repeated], axis=2)
        # (K, 10, 8)
        mask = tf.tile(mask, [1, 1, 2*self.units])
        concatenated = tf.multiply(concatenated, tf.cast(mask, tf.float32))
        # (K2, 10, 8)
        return concatenated


class FeatureNet(object):

    def __init__(self, training, batch_size):
        super(FeatureNet, self).__init__()
        self.training = training
        self.batch_size = batch_size
        self.feature = tf.placeholder(
            tf.float32, [None, 10, 7], name="feature"
        )
        self.number = tf.placeholder(
            tf.int64, [None], name="number"
        )
        self.coordinate = tf.placeholder(
            tf.int64, [None, 4], name="coordinate"
        )

        self.vfe1 = VFELayer(32)
        # self.vfe2 = VFELayer(32)
        self.dense = tf.layers.Dense(
            32, tf.nn.relu
        )
        self.batch_normal = tf.layers.BatchNormalization()

        mask = tf.not_equal(tf.reduce_max(self.feature, axis=2, keep_dims=True), 0)
        x = self.vfe1.apply(self.feature, mask, self.training)
        # x = self.vfe2.apply(x, mask, self.training)
        # (None, 10, 32)
        x = self.dense.apply(x)
        x = self.batch_normal.apply(x, self.training)
        # (None, 10, 32)
        voxelwise = tf.reduce_max(x, axis=1)

        """
        self.coordinate (None, 4)
        voxelwise (None, 32)
        [5, 10, 10, 10, 32]
        """
        print(self.coordinate.shape)
        print(voxelwise.shape)
        self.outputs = tf.scatter_nd(
            indices=self.coordinate, updates=voxelwise, shape=[self.batch_size, 10, 10, 10, 32])


def build_input(voxel_dict_list):
    batch_size = len(voxel_dict_list)

    feature_list = []
    number_list = []
    coordinate_list = []

    for i, voxel_dict in zip(range(batch_size), voxel_dict_list):
        feature_list.append(voxel_dict['feature_buffer'])
        number_list.append(voxel_dict['number_buffer'])
        coordinate = voxel_dict['coordinate_buffer']
        coordinate_list.append(
            np.pad(coordinate, ((0, 0), (1, 0)), mode="constant", constant_values=i)
        )

    feature = np.concatenate(feature_list)
    number = np.concatenate(number_list)
    coordinate = np.concatenate(coordinate_list)
    # print("coordinate : ", coordinate.shape)
    return batch_size, feature, number, coordinate


def run(batch_size, feature, number, coordinate):
    with tf.Session() as sess:
        model = FeatureNet(training=False, batch_size=batch_size)
        tf.global_variables_initializer().run()
        for i in range(10):
            time_start = time.time()
            feed = {
                model.feature: feature,
                model.number: number,
                model.coordinate: coordinate
            }
            outputs = sess.run([model.outputs], feed)
            print(outputs[0].shape)
            time_end = time.time()
            # print(time_end-time_start)


def main():
    data_dir = 'object_voxel'
    batch_size = 5

    filelist = [f for f in os.listdir(data_dir)]

    voxel_dict_list = []
    for id in range(0, len(filelist), batch_size):
        pre_time = time.time()
        batch_file = [f for f in filelist[id: id+batch_size]]
        voxel_dict_list = []
        for file in batch_file:
            voxel_dict_list.append(np.load(os.path.join(data_dir, file)))

        batch_size, feature, number, coordinate = build_input(voxel_dict_list)
        # print(time.time() - pre_time)

    run(batch_size, feature, number, coordinate)


if __name__ == '__main__':
    main()
