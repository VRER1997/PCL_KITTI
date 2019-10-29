import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import mayavi.mlab
from matplotlib.ticker import MultipleLocator


def plot3d(data):
    if data.shape[1] != 3:
        print("data should have x, y, z !!!")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2])
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_zticks([0.2, 0.4, 0.6, 0.8, 1.0])
    plt.show()


def plot_pointcloud(data):
    if data.shape[1] != 3:
        print("data should have x, y, z !!!")
        return
    fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 260))
    mayavi.mlab.points3d(data[:, 0], data[:, 1], data[:, 2], data[:, 2], mode="point", colormap="spectral",
                         figure=fig)
    mayavi.mlab.show()
