from scipy.spatial.transform import Rotation
import numpy as np


def interp_vector(a, b, n):
    delta = (b - a) / (n - 1)
    list_vecs = []
    for i in range(n):
        new_vec = a + delta * i
        list_vecs.append(new_vec)
    return np.asarray(list_vecs)


def random_sample(value_range):
    return (value_range[1] - value_range[0]) * np.random.random() + value_range[0]


def polar_translation(r, theta):
    # follow math convention for polar coordinates
    # r: radius
    # theta: azimuth (horizontal)
    # psi: vertical
    theta = np.pi / 2 + theta  # azimuth is measured from x-axis but onboard camera points down orthogonal axis, so shifting by 90 degrees to get range in camera's fov
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def convert_t_body_2_world(relative_pos, pos_x, pos_y, pos_z, q):
    rotation = Rotation.from_quat(q)
    # apply inverse of tracker's rotation to relative pos, so it is back on world axes
    rotated_pos = rotation.apply(relative_pos, inverse=True)
    # sum the two positions so that world origin is origin
    world_pos = [pos_x, pos_y, pos_z] + rotated_pos
    return world_pos


def get_q_as_yaw(q):
    rotation = Rotation.from_quat(q)
    euler_angles = rotation.as_euler('zyx', degrees=True)
    return euler_angles[0]  # returning yaw value, which is in place z because scipy defines z-axis as vertical axis


def get_yaw_as_q(yaw):
    rot = Rotation.from_euler('zyx', [yaw, 0, 0], degrees=True)  # NB yaw in place z because scipy defines z-axis as vertical axis
    rot_q = rot.as_quat(False)  # returns numpy array in format (x,y,z,w), again with z referring to rotation around vertical axis
    return rot_q
