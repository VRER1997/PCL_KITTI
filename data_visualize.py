# -*- coding: utf-8 -*-
# @Time    : 2019/10/22 上午9:11
# @Author  : Gao Xiaosa

import os
import numpy as np
from tools import plot3d

data_dir = './object_cloud/training'
filelist = [f for f in os.listdir(data_dir)]
ff = []
for file in filelist:
    if file == '000955_0_1.txt':
        ff.append(file)

for file in ff:
    print(file)
    data = np.loadtxt(os.path.join(data_dir, file), dtype=np.float32)
    name, extension = os.path.splitext(file)
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    f = name + " " + str(data.shape) + " " + str(np.max(x) - np.min(x)) + \
        " " + str(np.mean(y) - np.min(y)) + " " + str(np.mean(z) - np.min(z))
    # np.savetxt('result.txt', f)
    print(f)
    plot3d(data[:, :3])

    """
    def norm(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x) + 1)

    x = norm(x).reshape(-1, 1)
    y = norm(y).reshape(-1, 1)
    z = norm(z).reshape(-1, 1)
    """

    data -= np.min(data, axis=0)
    size = np.max(data, axis=0) - np.min(data, axis=0)

    ## 缩小至1x1x1 的包围框中
    maxl = np.max(size[:3])
    rate = 1.0 / maxl
    data *= rate

    ## 中心化
    data += 0.5 - size * rate * 0.5

    # normalized = np.hstack([x,y,z])
    # print(normalized.shape)
    #
    plot3d(data[:, :3])
