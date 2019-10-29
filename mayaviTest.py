import numpy as np
from Calibration import Calibration

from mayavi import mlab

data_dir = '/home/jlurobot/Documents/PointRCNN/data/KITTI/object/training/'


def get_corners(height, width, length, x, y, z, θ, rotation=True):
    corners = np.array(
        [[-length / 2, -length / 2, length / 2, length / 2, -length / 2, -length / 2, length / 2, length / 2],
         [width / 2, -width / 2, -width / 2, width / 2, width / 2, -width / 2, -width / 2, width / 2],
         [0, 0, 0, 0, height, height, height, height]])

    rotMat = np.array([[np.cos(θ), -np.sin(θ), 0],
                       [np.sin(θ), np.cos(θ), 0],
                       [0, 0, 1]])
    if rotation:
        cornersPos = (np.dot(rotMat, corners) + np.tile([x, y, z], (8, 1)).T).transpose()
        corner1, corner2, corner3, corner4, corner5, corner6, corner7, corner8 = cornersPos[0], cornersPos[1], \
                                                                                 cornersPos[2], cornersPos[3], \
                                                                                 cornersPos[4], cornersPos[5], \
                                                                                 cornersPos[6], cornersPos[7]
    else:
        cornersPos = (corners + np.tile([x, y, z], (8, 1)).T).transpose()
        corner1, corner2, corner3, corner4, corner5, corner6, corner7, corner8 = cornersPos[0], cornersPos[1], \
                                                                                 cornersPos[2], cornersPos[3], \
                                                                                 cornersPos[4], cornersPos[5], \
                                                                                 cornersPos[6], cornersPos[7]

    return list(corner1), list(corner2), list(corner3), list(corner4), list(corner5), list(corner6), list(
        corner7), list(corner8)


def draw_gt_boxes3d(gt_boxes3d, fig, thres = 0, color=(1,1,1), line_width=1, draw_text=True, text_scale=(1,1,1), color_list=None):

    num = len(gt_boxes3d)

    for n in range(num):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n]
        if n >= thres:
            color = (0, 1, 0)
        else:
            color = (1, 0, 0)
        if draw_text: mlab.text3d(b[4,0], b[4,1], b[4,2], '%d'%n, scale=text_scale, color=color, figure=fig)
        for k in range(0,4):
            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j = k, (k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j, 1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+1)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)
    mlab.show()
    return fig


def get_kitti_object_cloud(img_id):

    # img_id = 955
    lidar_path = data_dir + "velodyne/%06d.bin" % img_id
    label_path = data_dir + "label_2/%06d.txt" % img_id
    calib_path = data_dir + "calib/%06d.txt" % img_id

    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)  # .astype(np.float16)
    lables = np.loadtxt(label_path,
                        dtype={'names': (
                        'type', 'truncated', 'occuluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'h', 'w', 'l', 'x',
                        'y', 'z', 'rotation_y'),
                               'formats': (
                               'S14', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',
                               'float', 'float', 'float', 'float', 'float')})

    calibs = Calibration(calib_path)

    if lables.size == 1:
        lables = lables[np.newaxis]

    """
    fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 260))
    mayavi.mlab.points3d(points[:, 0], points[:, 1], points[:, 2], points[:, 2], mode="point", colormap="spectral", figure=fig)
    mayavi.mlab.title("before")
    mayavi.mlab.show()
    """

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
            filt = np.logical_and(x_filt, y_filt)
            filt = np.logical_and(filt, z_filt)

            col = np.zeros(shape=(points.shape[0], 1))

            points = np.hstack([points, col])

            # object_cloud = points[filt, :]  # 过滤
            print(col.shape)
            print(points.shape)

            for j in range(len(col)):
                if filt[j]:
                    points[j, -1] = 200
                else:
                    points[j, -1] = points[j, 2]

            print(label['type'])

            # 形成包围框的上四个点
            centour = []
            centour.append([min_point_AABB[0], min_point_AABB[1], max_point_AABB[2], 200, 200])
            centour.append([min_point_AABB[0], max_point_AABB[1], max_point_AABB[2], 200, 200])
            centour.append([max_point_AABB[0], min_point_AABB[1], max_point_AABB[2], 200, 200])
            centour.append([max_point_AABB[0], max_point_AABB[1], max_point_AABB[2], 200, 200])

            points = np.vstack([points, centour])

            fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 260))
            mlab.points3d(points[:, 0], points[:, 1], points[:, 2], points[:, -1], mode="point", colormap="spectral",
                       figure=fig)
            mlab.show()

for i in range(8, 15):
    get_kitti_object_cloud(i)


"""

box_gt = []
for label in lables:
    if label['type'] not in [b'Car', b'Pedestrian', b'Cyclist']:
        continue
    height, width, length, x_tmp, y_tmp, z_tmp, θ = label['h'], label['w'], label['l'], label['x'], label['y'], label['z'], label['rotation_y']
    x, y, z, θ = z_tmp, -x_tmp, y_tmp - VELODYNE_HEIGHT, np.pi / 2 - θ
    C1, C2, C3, C4, C5, C6, C7, C8 = get_corners(height, width, length, x, y, z, θ, rotation=True)
    box_gt.append(list((get_corners(height, width, length, x, y, z, θ, rotation=True))))

print(box_gt)

all_boxes = np.concatenate((box_gt,))

fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 260))
mlab.points3d(points[:, 0], points[:, 1], points[:, 2], points[:, -1], mode="point", colormap="spectral", figure=fig)
fig = draw_gt_boxes3d(all_boxes, fig, thres=len(box_gt), color=(0, 1, 0))
mlab.show()

"""