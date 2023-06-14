import numpy as np


def rotate_by_angle(v, theta_x, theta_y, theta_z):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta_x), -np.sin(theta_x)],
                    [0, np.sin(theta_x), np.cos(theta_x)]])
    R_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                    [0, 1, 0],
                    [-np.sin(theta_y), 0, np.cos(theta_y)]])
    R_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                    [np.sin(theta_z), np.cos(theta_z), 0],
                    [0, 0, 1]])

    v = np.expand_dims(v, -1)
    new_v = np.matmul(R_z, np.matmul(R_y, (np.matmul(R_x, v))))
    rot_mat = np.matmul(R_z, np.matmul(R_y, R_x))
    return new_v.squeeze(-1), rot_mat


def rot_mat_by_angle(theta_x, theta_y, theta_z):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta_x), -np.sin(theta_x)],
                    [0, np.sin(theta_x), np.cos(theta_x)]])
    R_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                    [0, 1, 0],
                    [-np.sin(theta_y), 0, np.cos(theta_y)]])
    R_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                    [np.sin(theta_z), np.cos(theta_z), 0],
                    [0, 0, 1]])

    rot_mat = np.matmul(R_z, np.matmul(R_y, R_x))
    return rot_mat


def random_rotate(object_points, hand_joints):
    x_angle = np.random.rand() * np.pi * 2.0
    y_angle = np.random.rand() * np.pi * 2.0
    z_angle = np.random.rand() * np.pi * 2.0

    rot_mat = rot_mat_by_angle(x_angle, y_angle, z_angle)
    new_obj_points = np.matmul(rot_mat, np.expand_dims(object_points, -1)).squeeze(-1)
    new_hand_joints = np.matmul(rot_mat, np.expand_dims(hand_joints, -1)).squeeze(-1)

    return new_obj_points, new_hand_joints, rot_mat
