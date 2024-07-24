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
    return (math.exp(k * diff) - 1) / (math.exp(k * max_diff) - 1)


# Function to be used where we want to penalise harshly immediately
def normalized_natural_log_impact(diff, max_diff, k=1):
    return math.log(k * diff + 1) / math.log(k * max_diff + 1) if (diff > 0) else 0

