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
