from enum import IntEnum


class EpisodeTerminationType(IntEnum):
    MAXIMUM_DISTANCE = 0
    TARGET_OUT_OF_VIEW = 1
    TARGET_COLLISION = 2
    THRESHOLD_REACHED = 3


class TrainingType(IntEnum):
    TRAINING = 0
    INFERENCE = 1
