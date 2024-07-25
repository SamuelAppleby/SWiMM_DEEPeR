import math

import numpy as np


def calc_metrics(rov_pos, rov_fwd, target_pos):
    # heading vector from rover to target
    heading = target_pos - rov_pos

    # normalize
    norm_heading = heading / np.linalg.norm(heading)

    # calculate radial distance on the flat y-plane
    raw_d = math.sqrt(math.pow(heading[0], 2) + math.pow(heading[2], 2))

    # calculate angle between rover's forward facing vector and heading vector
    dot = np.dot(norm_heading, rov_fwd)

    # floating-point inaccuracy may cause epsilon violations, so clamp to legal values
    dot = np.clip(dot, -1, 1)
    acos = np.arccos(dot)

    a = np.degrees(acos)

    assert not math.isnan(a)

    return raw_d, a


# Function to be used where we want to penalise harshly towards to extremities
def normalized_exponential_impact(diff, max_diff, k=1):
    assert diff >= 0, f'The difference {diff} cannot be less than 0'
    out_of_bounds = (diff >= max_diff)

    if out_of_bounds:
        diff = np.clip(diff, 0, max_diff)       # If the rover is too far away, clip the difference

    error = (math.exp(k * diff) - 1) / (math.exp(k * max_diff) - 1)
    assert 1 >= error >= 0, f'The error {error} must be between 0 and 1'
    return error, out_of_bounds


# Function to be used where we want to penalise harshly immediately
def normalized_natural_log_impact(diff, max_diff, k=1):
    assert diff >= 0, f'The difference {diff} cannot be less than 0'        # We are allowing diff > max_diff as the dolphin can still be in view (we use the central position of the mesh)
    out_of_bounds = (diff >= max_diff)      # We don't want to return this as the dolphin may still be in view

    if out_of_bounds:
        diff = np.clip(diff, 0, max_diff)

    error = math.log(k * diff + 1) / math.log(k * max_diff + 1) if (diff > 0) else 0
    assert 1 >= error >= 0, f'The error {error} must be between 0 and 1'
    return error, out_of_bounds

