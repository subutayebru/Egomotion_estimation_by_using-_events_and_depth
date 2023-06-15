import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.spatial.transform import Rotation as R



def plotPose(X, Y, Z, oriX, oriY, oriZ, path_prefix, file_name):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(X, Y, Z, cmap='jet', c=t, marker='.', alpha=1)

    max_axis_range = max(np.max(X) - np.min(X), np.max(Y) - np.min(Y), np.max(Z) - np.min(Z))
    i = 0
    a = max_axis_range / 20
    for i in range(0, X.shape[0], 100):
        ax.plot([X[i], X[i] + a * oriX[i, 0]], [Y[i], Y[i] + a * oriX[i, 1]], [Z[i], Z[i] + a * oriX[i, 2]], color='r',
                linestyle='-', linewidth=2)
        ax.plot([X[i], X[i] + a * oriY[i, 0]], [Y[i], Y[i] + a * oriY[i, 1]], [Z[i], Z[i] + a * oriY[i, 2]], color='g',
                linestyle='-', linewidth=2)
        ax.plot([X[i], X[i] + a * oriZ[i, 0]], [Y[i], Y[i] + a * oriZ[i, 1]], [Z[i], Z[i] + a * oriZ[i, 2]], color='b',
                linestyle='-', linewidth=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_zlim3d([-10, 10])
    ax.auto_scale_xyz([np.min(X), np.min(X) + max_axis_range], [np.min(Y), np.min(Y) + max_axis_range],
                      [np.min(Z), np.min(Z) + max_axis_range])

    plt.show()


path_prefix = path_prefix
file_name = file_name
pose = np.loadtxt(path_prefix + file_name , delimiter=' ') 
start_t = 0
stop_t = pose[-1, 0]
# T_lidar_1 = np.array([-0.0078031, 0.817209, 0.541196, 0.917704,
#                       -0.117454, -0.572388, 0.727834, 0.48356,
#                       0.904633, -0.0673863, 0.100153, -0.665694,
#                       0, 0, 0, 1]).reshape((4, 4))
T_lidar_0 = np.array([-0.3819, -0.357824, 0.852124, 0.614092, 
                      -0.847004, -0.233392, -0.477611, -0.322554,
                      0.36978, -0.904152, -0.213946, -0.0494612,
                      0, 0, 0, 1]).reshape((4, 4))
euler_center_baumer = np.array([-89.90413419510571,
                                -0.24585344685304605,
                                -64.33530496116742])
t_center_baumer = np.array([1.4970588815871406,
                            0.4157759017823685,
                            1.2825050304352636])
R_center_baumer = R.from_euler('xyz', euler_center_baumer).as_matrix()

T_center_baumer = np.eye(4)
T_center_baumer[:3, :3] = R_center_baumer
T_center_baumer[:3, 3] = t_center_baumer

T_lidar_baumer = np.array([0.06527, -0.344349, 0.93657, 0.947059,
                           -0.967731, -0.250765, -0.0247573, 0.415776,
                           0.243384, -0.904732, -0.349605, -0.452495,
                           0, 0, 0, 1]).reshape((4, 4))
T_baumer_1 = np.array([0.9101440571262656, -0.05112656051741041, 0.41112512704455406, -0.216258895905397,
                       0.04647405104554883, 0.9986921401911144, 0.021311304509745152, -0.05954830294291894,
                       -0.41167700671419194, -0.00028970700877638776, 0.9113297746769236, 0.024422175420170375,
                       0, 0, 0, 1]).reshape((4,4))
T_baumer_1[:3, :3] = np.eye(3)
rotvec = R.from_matrix(T_lidar_baumer[:3, :3]).as_rotvec()
T_hand_eye = np.identity(4) 
print(T_center_baumer.astype(np.float64))
print(T_hand_eye.astype(np.float64))

# path_prefix = '/home/suman/data/rpg/DSEC/zurich_city_04-odometry/'
#
# pos_fname = path_prefix + 'position.npy'
# pos = np.load(pos_fname)
#
# ori_fname = path_prefix + 'orientation.npy'
# ori = np.load(ori_fname)
#
# pos_time_fname = path_prefix + 'time.npy'
# pos_time = np.load(pos_time_fname)

# # time of DSEC zurich02b acc to the left events
# start_t = 56246.41
# stop_t = 56307.61
# start_t = pos_time[0]/1e6
# stop_t = pos_time[-1]/1e6
# start_id = np.searchsorted(pos_time, start_t * 1e6)
# stop_id = np.searchsorted(pos_time, stop_t * 1e6)
# T_hand_eye = np.array([00.00147698, -0.00935589, 0000.999955, 0000.434805,
#                        00 - 0.999689, 000.0248923, 00.00170949, 0000.298816,
#                        0 - 0.0249072, 00 - 0.999646, -0.00931621, 00 - 0.214341,
#                        00000000000, 00000000000, 00000000000, 1]).reshape((4, 4))

# time of DSEC zurich04a acc to the left events
# start_t = 36470.60
# stop_t = 36505.59
# start_t = pos_time[0]/1e6
# stop_t = pos_time[-1]/1e6

# tum-vie calibA
# T_hand_eye = np.array([0000 - 0.93131, 00000.123631, 000 - 0.342603, 00 - 0.0204617,
#                        00000.364226, 00000.313839, 000 - 0.876838, 00 - 0.0580755,
#                        -0.000882187, 000 - 0.941393, 000 - 0.337311, 000 - 0.130939,
#                        000000000000, 000000000000, 000000000000, 1]).reshape((4, 4))

# tum-vie calib-B
# T_hand_eye = np.array([00 - 0.516976, 00 - 0.295714, 0000.803299, 0000.058292,
#                        00 - 0.855946, 0000.189111, 00 - 0.481242, 0 - 0.0335537,
#                        -0.00960316, 00 - 0.936371, 00 - 0.350881, 00 - 0.134856,
#                        00000000000, 00000000000, 00000000000, 1]).reshape((4, 4))
# path_prefix = '/home/suman/data/data_ARA/Data ARA/optitrack/hb1/objPose_2021-09-29-12-39-53/groundtruth.txt'
# path_prefix = '/mnt/HD4TB/data/tum-vie/mocap-6dof-vi_gt_data/mocap_data.txt'

# path_prefix += 'pose.txt'
# start_t = (start_t - 36339.513459)
# stop_t = (stop_t - 36339.513459)
# pose = np.loadtxt(path_prefix)
pos_time = pose[:, 0]
pos = pose[:, 1:4]
ori = pose[:, 4:]

start_id = np.searchsorted(pos_time, start_t)
stop_id = np.searchsorted(pos_time, stop_t)

start_id = 1
stop_id = 60
stop_id = pos_time.shape[0]

X = pos[start_id:stop_id, 0]
Y = pos[start_id:stop_id, 1]
Z = pos[start_id:stop_id, 2]
t = pos_time[start_id: stop_id]

ori_section = ori[start_id:stop_id, :]
unitX = np.array([1, 0, 0])
unitY = np.array([0, 1, 0])
unitZ = np.array([0, 0, 1])

rotation_matrices = (R.from_quat(ori_section)).as_matrix()
transformation_matrices = np.zeros((rotation_matrices.shape[0], 4, 4))
for j in range(rotation_matrices.shape[0]):
    rotation_matrices[j, :, :] = rotation_matrices[j, :, :] @ T_hand_eye[:3, :3]

oriX = rotation_matrices[:, :, 0]
oriY = rotation_matrices[:, :, 1]
oriZ = rotation_matrices[:, :, 2]

plotPose(X, Y, Z, oriX, oriY, oriZ, '/home/bru/Dev/DATA/scn2_take01/take01/', 'gt.csv')
