# -*- coding: utf-8 -*-
# @Time    : 2019/10/22 下午7:47
# @Author  : Gao Xiaosa


import numpy as np
import os
from multiprocessing import Queue, Process, Value
import threading
import time
from processObject import process_pointcloud


class DataLoader(object):

    def __init__(self,
                 object_dir='object_cloud/training',
                 queue_size=20,
                 require_shuffle=False,
                 is_testset=True,
                 batch_size=1,
                 use_multi_process_num=0):
        self.object_dir = object_dir
        self.queue_size = queue_size
        self.is_testset = is_testset
        self.required_shuffle = require_shuffle if not self.is_testset else False
        self.use_multi_process_num = use_multi_process_num if not self.is_testset else 1
        self.batch_size = batch_size

        self.f_object = [f for f in os.listdir(self.object_dir) if f.endswith('txt')]
        print(len(self.f_object))
        #避免　类别不平衡
        self.sampling()
        print(len(self.f_object))
        # shuffle ?
        if require_shuffle:
            np.random.shuffle(self.f_object)

        self.data_tag = []

        for name in self.f_object:
            if name[7] == '1':
                self.data_tag.append([0, 1])
            else:
                self.data_tag.append([1, 0])

        self.dataset_size = len(self.f_object)
        self.already_extract_data = 0

        self.dataset_queue = Queue()
        self.queue_size = queue_size
        self.load_index = 0

        if self.use_multi_process_num == 0:
            self.loader_worker = [threading.Thread(target=self.load_worker_main, args=(self.batch_size,))]
        else:
            self.loader_worker = [Process(target=self.load_worker_main, args=(self.batch_size,)) \
                                  for i in range(self.use_multi_process_num)]

        self.work_exit = Value('i', 0)
        [i.start() for i in self.loader_worker]

        self.load_index = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.work_exit.value = True

    def __len__(self):
        return self.dataset_size

    def load_worker_main(self, batch_size):
        while not self.work_exit.value:
            if self.dataset_queue.qsize() >= self.queue_size // 2:
                time.sleep(1)
            else:
                self.fill_queue(batch_size)

    def fill_queue(self, batch_size):
        load_index = self.load_index
        self.load_index += batch_size
        if self.load_index >= self.dataset_size:
            if not self.is_testset:
                load_index = 0
                self.load_index = load_index + batch_size
            else:
                self.work_exit.value = True

        tag, voxel, object_cloud = [], [], []
        for _ in range(batch_size):
            try:
                object_cloud.append(np.loadtxt(os.path.join(self.object_dir, self.f_object[load_index]),
                                                dtype=np.float32).reshape((-1, 4)))
                tag.append(self.data_tag[load_index])
                voxel.append(process_pointcloud(object_cloud[-1]))
                load_index += 1
            except:
                if not self.is_testset:
                    self.load_index = 0
                else:
                    self.work_exit.value = True

        _, vox_feature, vox_number, vox_coordinate = build_input(voxel)
        self.dataset_queue.put_nowait(
            (tag, (vox_feature, vox_number, vox_coordinate))
        )

    def load(self):
        try:
            if self.is_testset and self.already_extract_data >= self.dataset_size:
                return None

            buff = self.dataset_queue.get()
            tag = buff[0]
            vox_feature = buff[1][0]
            vox_number = buff[1][1]
            vox_coordinate = buff[1][2]

            self.already_extract_data += self.batch_size

            ret = (
                np.array(tag),
                np.array(vox_feature),
                np.array(vox_number),
                np.array(vox_coordinate)
            )
        except:
            print('Dataset empty')
            ret = None

        return ret

    def get_shape(self):
        return self.f_object.shape

    def sampling(self):
        car, walker, waker_num = [], [], 0
        for name in self.f_object:
            if name[7] == '1':
                waker_num += 1
                walker.append(name)
            else:
                car.append(name)

        car = np.random.permutation(car)[:waker_num]
        self.f_object = np.hstack([car, np.array(walker)])


def build_input(voxel_dict_list):
    batch_size = len(voxel_dict_list)

    feature_list = []
    number_list = []
    coordinate_list = []

    for i, voxel_dict in zip(range(batch_size), voxel_dict_list):
        feature_list.append(voxel_dict['feature_buffer'])
        number_list.append(voxel_dict['number_buffer'])
        coordinate = voxel_dict['coordinate_buffer']
        coordinate_list.append(np.pad(coordinate, ((0, 0), (1, 0)), mode='constant', constant_values=i))

    # print("len feature_list :", len(feature_list))
    feature = np.concatenate(feature_list)
    number = np.concatenate(number_list)
    coordinate = np.concatenate(coordinate_list)

    return batch_size, feature, number, coordinate


if __name__ == '__main__':
    d = DataLoader(object_dir="object_cloud/training",
                   queue_size=10,
                   is_testset=False,
                   batch_size=10,
                   )
    q = d.load()
    print(len(q))