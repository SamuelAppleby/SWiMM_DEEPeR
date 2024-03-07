from enum import IntEnum


class Protocol(IntEnum):
    UDP = 0
    TCP = 1

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class EpisodeTerminationType(IntEnum):
    MAXIMUM_DISTANCE = 0
    TARGET_OUT_OF_VIEW = 1
    TARGET_COLLISION = 2
    THRESHOLD_REACHED = 3

class TrainingType(IntEnum):
    TRAINING = 0
    INFERENCE = 1
