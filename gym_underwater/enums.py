from enum import IntEnum, Enum


class EpisodeTerminationType(IntEnum):
    MAXIMUM_DISTANCE = 0
    TARGET_OUT_OF_VIEW = 1
    TARGET_COLLISION = 2
    THRESHOLD_REACHED = 3


class TrainingType(IntEnum):
    TRAINING = 0
    INFERENCE = 1

    def __str__(self):
        if self == TrainingType.TRAINING:
            return "training"
        elif self == TrainingType.INFERENCE:
            return "inference"
        return super().__str__()


class ObservationType(Enum):
    IMAGE = 'image'
    VECTOR = 'vector'
    CMVAE = 'cmvae'
