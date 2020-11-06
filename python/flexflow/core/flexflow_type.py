from enum import Enum

class ActiMode(Enum):
  AC_MODE_NONE = 10
  AC_MODE_RELU = 11
  AC_MODE_SIGMOID = 12
  AC_MODE_TANH = 13

class AggrMode(Enum):
  AGGR_MODE_NONE = 20
  AGGR_MODE_SUM = 21
  AGGR_MODE_AVG = 22

class PoolType(Enum):
  POOL_MAX = 30
  POOL_AVG = 31

class DataType(Enum):
  DT_FLOAT = 40
  DT_DOUBLE = 41
  DT_INT32 = 42
  DT_INT64 = 43
  DT_BOOLEAN = 44

class LossType(Enum):
  LOSS_CATEGORICAL_CROSSENTROPY = 50
  LOSS_SPARSE_CATEGORICAL_CROSSENTROPY = 51
  LOSS_MEAN_SQUARED_ERROR_AVG_REDUCE = 52
  LOSS_MEAN_SQUARED_ERROR_SUM_REDUCE = 53

class MetricsType(Enum):
  METRICS_ACCURACY = 1001
  METRICS_CATEGORICAL_CROSSENTROPY = 1002
  METRICS_SPARSE_CATEGORICAL_CROSSENTROPY = 1004
  METRICS_MEAN_SQUARED_ERROR = 1008
  METRICS_ROOT_MEAN_SQUARED_ERROR = 1016
  METRICS_MEAN_ABSOLUTE_ERROR=1032

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
  OUTPUT = 2050

def enum_to_int(enum, enum_item):
  for item in enum:
    if (enum_item == item):
      return item.value

  print(enum_item)
  print(enum)
  assert 0, "unknow enum type " + str(enum_item) + " " + str(enum)
  return -1

def int_to_enum(enum, value):
  for item in enum:
    if (item.value == value):
      return item

  assert 0, "unknow enum value " + str(value) + " " + str(enum)
