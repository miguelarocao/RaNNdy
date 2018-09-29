from enum import Enum, unique

@unique
class DataSetType(Enum):
    TRAIN = 1
    VALIDATION = 2
    TEST = 3

@unique
class MetricType(Enum):
    TOTAL_LOSS = 1
    RECONSTRUCTION_LOSS = 2
    KL_LOSS = 3
    BLEU = 4