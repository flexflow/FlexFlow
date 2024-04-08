from .config import *

from enum import Enum


class ActiMode(Enum):
    AC_MODE_NONE = 10
    AC_MODE_RELU = 11
    AC_MODE_SIGMOID = 12
    AC_MODE_TANH = 13
    AC_MODE_GELU = 14


class RegularizerMode(Enum):
    REG_MODE_NONE = 17
    REG_MODE_L1 = 18
    REG_MODE_L2 = 19


class AggrMode(Enum):
    AGGR_MODE_NONE = 20
    AGGR_MODE_SUM = 21
    AGGR_MODE_AVG = 22


class PoolType(Enum):
    POOL_MAX = 30
    POOL_AVG = 31


class DataType(Enum):
    DT_BOOLEAN = 40
    DT_INT32 = 41
    DT_INT64 = 42
    DT_HALF = 43
    DT_FLOAT = 44
    DT_DOUBLE = 45
    DT_NONE = 49


class LossType(Enum):
    LOSS_CATEGORICAL_CROSSENTROPY = 50
    LOSS_SPARSE_CATEGORICAL_CROSSENTROPY = 51
    LOSS_MEAN_SQUARED_ERROR_AVG_REDUCE = 52
    LOSS_MEAN_SQUARED_ERROR_SUM_REDUCE = 53
    LOSS_IDENTITY = 54


class CompMode(Enum):
    TRAINING = 70
    INFERENCE = 71


class ParameterSyncType(Enum):
    NONE = 80
    PS = 81
    NCCL = 82


class MetricsType(Enum):
    METRICS_ACCURACY = 1001
    METRICS_CATEGORICAL_CROSSENTROPY = 1002
    METRICS_SPARSE_CATEGORICAL_CROSSENTROPY = 1004
    METRICS_MEAN_SQUARED_ERROR = 1008
    METRICS_ROOT_MEAN_SQUARED_ERROR = 1016
    METRICS_MEAN_ABSOLUTE_ERROR = 1032


class InferenceMode(Enum):
    INC_DECODING_MODE = 2001
    BEAM_SEARCH_MODE = 2002
    TREE_VERIFY_MODE = 2003


class ModelType(Enum):
    UNKNOWN = 3001
    LLAMA = 3002
    OPT = 3003
    FALCON = 3004
    STARCODER = 3005
    MPT = 3006


class OpType(Enum):
    CONV2D = 2011
    EMBEDDING = 2012
    POOL2D = 2013
    LINEAR = 2014
    SOFTMAX = 2015
    CONCAT = 2016
    FLAT = 2017
    MSELOSS = 2020
    BATCH_NORM = 2021
    RELU = 2022
    SIGMOID = 2023
    TANH = 2024
    ELU = 2025
    DROPOUT = 2026
    BATCH_MATMUL = 2027
    SPLIT = 2028
    RESHAPE = 2029
    TRANSPOSE = 2030
    REVERSE = 2031
    EXP = 2040
    ADD = 2041
    SUBTRACT = 2042
    MULTIPLY = 2043
    DIVIDE = 2044
    POW = 2045
    MEAN = 2046
    RSQRT = 2047
    SIN = 2048
    COS = 2049
    INPUT = 2050
    OUTPUT = 2051
    REDUCE_SUM = 2052
    MAX = 2053
    MIN = 2054
    MULTIHEAD_ATTENTION = 2060
    INC_MULTIHEAD_ATTENTION = 2061
    SPEC_INC_MULTIHEAD_SELF_ATTENTION = 2062
    TREE_INC_MULTIHEAD_SELF_ATTENTION = 2063
    SAMPLING = 2065
    ARGMAX = 2066
    GETITEM = 2070
    GETATTR = 2080
    EXPAND = 2081
    LAYER_NORM = 2082
    FLOOR_DIVIDE = 2083
    IDENTITY = 2084
    GELU = 2085
    PERMUTE = 2086
    SCALAR_MULTIPLY = 2087
    SCALAR_FLOORDIV = 2088
    SCALAR_ADD = 2089
    SCALAR_SUB = 2090
    SCALAR_TRUEDIV = 2091
    INIT_PARAM = 2092
    FLOAT = 2100
    CONTIGUOUS = 2101
    TO = 2102
    UNSQUEEZE = 2103
    TYPE_AS = 2104
    VIEW = 2105
    GATHER = 2106
    ATTRIBUTE = 2200
    RMS_NORM = 2300
    ARG_TOPK = 2301
    BEAM_TOPK = 2302
    ADD_BIAS_RESIDUAL_LAYERNORM = 2303
    SIGMOID_SILU_MULTI = 2304
    RESIDUAL_RMS_NORM = 2305
    RESIDUAL_LAYERNORM = 2306

class RequestType(Enum):
    REQ_INFERENCE = 4001
    REQ_FINETUNING = 4002

def enum_to_int(enum, enum_item):
    for item in enum:
        if enum_item == item:
            return item.value

    print(enum_item)
    print(enum)
    assert 0, "unknown enum type " + str(enum_item) + " " + str(enum)
    return -1


def int_to_enum(enum, value):
    for item in enum:
        if item.value == value:
            return item

    assert 0, "unknown enum value " + str(value) + " " + str(enum)


def enum_to_str(enum, enum_item):
    name = enum(enum_item).name
    return name


def str_to_enum(enum, value):
    for item in enum:
        if item.name == value:
            return item

    assert 0, "unknown enum value " + value + " " + str(enum)
