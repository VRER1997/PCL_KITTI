import numpy as np
import os
import tensorflow as tf
import multiprocessing

base_path = 'object_cloud/testing'
output_dir = 'object_voxel'
os.makedirs(output_dir, exist_ok=True)


def pointcloud_transform(data):

    data -= np.min(data, axis=0)
    size = np.max(data, axis=0) - np.min(data, axis=0)

    ## 缩小至1x1x1 的包围框中
    maxl = np.max(size[:3])
    rate = 1.0 / maxl
    data *= rate

    ## 中心化
    data += 0.5 - size * rate * 0.5

    return data


# [4, N] -> [voxel_dict]
def process_pointcloud(data):

    # if cls == "Car":
    #     voxel_size = np.array([0.4, 0.2, 0.2], dtype=tf.float32)
    #     lidar_coord = np.array([0, 40, 3], dtype=tf.float32)
    #     max_point_number = 35
    # else:
    #     voxel_size = np.array([0.4, 0.2, 0.2])
    #     lidar_coord = np.array([0, 20, 3], dtype=tf.float32)
    #     max_point_number = 45
    # scene_size = np.array([1, 1, 1], dtype=tf.float32)
    voxel_size = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    # grid_size = np.array([10, 10, 10], dtype=tf.float32)
    max_point_number = 10

    # np.random.shuffle(data)

    shifted_coord = data[:, :3] #+ lidar_coord
    # [x, y, z] -> [z, y, x]
    voxel_index = np.floor(
        shifted_coord[:, ::-1] / voxel_size
    ).astype(np.int)

    """
    print("before", len(voxel_index))

    filt1 = np.logical_and(voxel_index[:, 0] < 10, voxel_index[:, 1] < 10)
    filt2 = np.logical_and(voxel_index[:, 0] < 10, voxel_index[:, 2] < 10)
    filt = np.logical_and(filt1, filt2)

    voxel_index = voxel_index[filt, :]
    print(len(voxel_index))
    """
    # voxel_index <(zi, yi, xi)>
    coordinate_buffer = np.unique(voxel_index, axis=0)
    k = len(coordinate_buffer)
    t = max_point_number

    number_buffer = np.zeros(shape=(k), dtype=np.int64)
    feature_buffer = np.zeros(shape=(k, t, 7), dtype=np.float32)

    index_buffer = {}
    for i in range(k):
        index_buffer[tuple(coordinate_buffer[i])] = i

    for voxel, point in zip(voxel_index, data):
        index = index_buffer[tuple(voxel)]
        number = number_buffer[index]
        if number < t:
            feature_buffer[index, number, :4] = point
            number_buffer[index] += 1

    # print(feature_buffer[:, :, :3].sum(axis=1, keepdims=True).shape)
    feature_buffer[:, :, -3:] = feature_buffer[:, :, :3] - \
        feature_buffer[:, :, :3].sum(axis=1, keepdims=True) / number_buffer.reshape(k, 1, 1)

    voxel_dict = {
        "feature_buffer": feature_buffer,
        "coordinate_buffer": coordinate_buffer,
        "number_buffer": number_buffer
    }
    return voxel_dict


def worker(filelist):
    for file in filelist:
        data = np.loadtxt(os.path.join(base_path, file), dtype=np.float32).reshape(-1, 4)
        print(file)
        print(data[0])
        # make transformation
        data = pointcloud_transform(data)

        name, extension = os.path.splitext(file)
        voxel_dict = process_pointcloud(data)
        # print(voxel_dict)
        np.savez_compressed(os.path.join(output_dir, name), **voxel_dict)


if __name__ == '__main__':

    filelist = [f for f in os.listdir(base_path) if f.endswith('txt')]
    # worker((filelist[0],))

    num_worker = 8
    pool = multiprocessing.Pool()
    for sublist in np.array_split(filelist, num_worker):
        pool.apply_async(func=worker, args=(sublist,))
    pool.close()
    pool.join()
