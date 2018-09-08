from enum import Enum, unique

@unique
class DataSetType(Enum):
    TRAIN = 1
    VALIDATION = 2
    TEST = 3
