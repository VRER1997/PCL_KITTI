import numpy as np
from Calibration import Calibration
import os
import random
import multiprocessing

save_object_cloud_path = "object_cloud/"
os.makedirs(save_object_cloud_path, exist_ok=True)
os.makedirs(save_object_cloud_path+"training", exist_ok=True)
os.makedirs(save_object_cloud_path+"testing", exist_ok=True)

data_dir = '/home/jlurobot/Documents/PointRCNN/data/KITTI/object/training/'


def get_kitti_object_cloud(img_id):

    lidar_path = data_dir + "velodyne/%06d.bin" % img_id
    label_path = data_dir + "label_2/%06d.txt" % img_id
    calib_path = data_dir + "calib/%06d.txt" % img_id

    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)  # .astype(np.float16)
    lables = np.loadtxt(label_path,
        dtype={'names': ('type', 'truncated', 'occuluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'h', 'w', 'l', 'x', 'y', 'z', 'rotation_y'),
        'formats': ('S14', 'float', 'float', 'float', 'float', 'float', 'float', 'float','float', 'float', 'float', 'float', 'float', 'float', 'float')})

    calibs = Calibration(calib_path)

    if lables.size == 1:
        lables = lables[np.newaxis]

    i = 0
    for label in lables:
        i += 1
        if label['type'] != b'DontCare':
            # 将图像坐标转换为激光点云坐标
            xyz = calibs.project_rect_to_velo(np.array([[label['x'], label['y'], label['z']]]))

            # 中心点
            x = xyz[0][0]
            y = xyz[0][1]
            z = xyz[0][2]

            # 包围盒，最近点和最远点
            min_point_AABB = [x - label['l'] / 2, y - label['w'] / 2, z, ]
            max_point_AABB = [x + label['l'] / 2, y + label['w'] / 2, z + label['h'], ]

            # 过滤该范围内的激光点
            x_filt = np.logical_and(
                (points[:, 0] > min_point_AABB[0]), (points[:, 0] < max_point_AABB[0]))
            y_filt = np.logical_and(
                    (points[:, 1] > min_point_AABB[1]), (points[:, 1] < max_point_AABB[1]))
            z_filt = np.logical_and(
                    (points[:, 2] > min_point_AABB[2]), (points[:, 2] < max_point_AABB[2]))
            filt = np.logical_and(x_filt, y_filt)  # 必须同时成立
            filt = np.logical_and(filt, z_filt)  # 必须同时成立

            object_cloud = points[filt, :]  # 过滤

            # car : 0
            # pedestrian: 1
            # Cyclist : 2
            # other: 3
            adjust_label = 0

            print(label['type'])

            if label['type'] in [b'Car']:
                adjust_label = 0
            elif label['type'] in [b'Pedestrian']:
                adjust_label = 1
            elif label['type'] in [b'Cyclist']:
                adjust_label = 2
            elif label['type'] in [b'Misc']:
                continue

            if object_cloud.shape[0] <= 50:
                print('filter failed...', img_id, adjust_label, i)
                continue
            # print(object_cloud)
            d = random.random()
            if d > 0.9:
                np.savetxt(save_object_cloud_path+'testing/%06d_%d_%d.txt' % (img_id, adjust_label, i), object_cloud)
            else:
                np.savetxt(save_object_cloud_path+'training/%06d_%d_%d.txt' % (img_id, adjust_label, i), object_cloud)
            # result.append(object_cloud)


def worker(img_id_list):
    for i in img_id_list:
        get_kitti_object_cloud(i)


if __name__ == '__main__':

    # data_total_number = 7481
    data_total_number = 2000
    num_worker = 16
    pool = multiprocessing.Pool()
    for sublist in np.array_split(range(data_total_number), num_worker):
        pool.apply_async(worker, args=(sublist,))

    pool.close()
    pool.join()

    car_num, pedestrian_num, cyclist_num, point_num_list = 0, 0, 0, []

    files = os.listdir(save_object_cloud_path + "training")
    for file in files:
        if file[7] == "0":
            car_num += 1
        elif file[7] == "1":
            pedestrian_num += 1
        elif file[7] == "2":
            cyclist_num += 1
        f = np.loadtxt(os.path.join(save_object_cloud_path + "training", file)).reshape(-1,4)
        point_num_list.append(f.shape[0])
    print("car_num is : {}".format(car_num))
    print("pedestrian_num is : {}".format(pedestrian_num))
    print("cyclist_num is : {}".format(cyclist_num))
    print("point_num is : {}".format(np.mean(point_num_list)))

    """
    2000 : car 4757 pedestrian: 758
    
    ALL:
    car_num is :  23615
    pedestrian_num is :  3716
    cyclist_num is :  1158
    point_num is : 548.8228088033978
    """