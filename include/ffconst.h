#ifndef _FLEXFLOW_CONST_H_
#define _FLEXFLOW_CONST_H_

enum ActiMode {
  AC_MODE_NONE = 10,
  AC_MODE_RELU = 11,
  AC_MODE_SIGMOID = 12,
  AC_MODE_TANH = 13,
};

enum AggrMode {
  AGGR_MODE_NONE = 20,
  AGGR_MODE_SUM = 21,
  AGGR_MODE_AVG = 22,
};

enum PoolType {
  POOL_MAX = 30,
  POOL_AVG = 31,
};

enum DataType {
  DT_FLOAT = 40,
  DT_DOUBLE = 41,
  DT_INT32 = 42,
  DT_INT64 = 43,
  DT_BOOLEAN = 44,
};

enum LossType {
  LOSS_CATEGORICAL_CROSSENTROPY = 50,
  LOSS_SPARSE_CATEGORICAL_CROSSENTROPY = 51,
  LOSS_MEAN_SQUARED_ERROR_AVG_REDUCE = 52,
  LOSS_MEAN_SQUARED_ERROR_SUM_REDUCE = 53,
};

enum MetricsType {
  METRICS_ACCURACY = 1001,
  METRICS_CATEGORICAL_CROSSENTROPY = 1002,
  METRICS_SPARSE_CATEGORICAL_CROSSENTROPY = 1004,
  METRICS_MEAN_SQUARED_ERROR = 1008,
  METRICS_ROOT_MEAN_SQUARED_ERROR = 1016,
  METRICS_MEAN_ABSOLUTE_ERROR = 1032,
};


// This is consistent with TASO's OpType
// https://github.com/jiazhihao/TASO/blob/master/include/taso/ops.h#L75-L138
enum OperatorType {
  OP_INPUT,
  OP_WEIGHT,
  OP_ANY,
  OP_CONV2D,
  OP_DROPOUT,
  OP_LINEAR,
  OP_POOL2D,
  OP_RELU,
  OP_SIGMOID,
  OP_TANH,
  OP_FLAT,
  OP_SOFTMAX,
  OP_BATCHNORM,
  OP_CONCAT,
  OP_SPLIT,
  OP_EMBEDDING,
  OP_ELEMENTWISE,
  OP_RESHAPE,
  OP_TRANSPOSE,
  OP_EW_ADD,
  OP_EW_MUL,
  OP_MATMUL,
  OP_MUL,
  OP_ENLARGE,
  OP_MERGE_GCONV,
  OP_CONSTANT_IMM,
  OP_CONSTANT_ICONV,
  OP_CONSTANT_ONE,
  OP_CONSTANT_POOL,
  OP_SQUEEZE, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Squeeze
  OP_UNSQUEEZE, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Unsqueeze
  OP_EW_SUB, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sub
  OP_EW_DIV, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Div
  OP_EW_EQUAL, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Equal
  OP_EW_GREATER, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Greater
  OP_EW_LESS, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Less
  OP_EW_MAX, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Max
  OP_EW_MIN, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Min
  OP_REDUCE_ARGMAX, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#ArgMax
  OP_REDUCE_ARGMIN, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#ArgMin
  OP_REDUCE_MAX, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMax
  OP_REDUCE_MEAN, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMean
  OP_REDUCE_MIN, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMin
  OP_REDUCE_PROD, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceProd
  OP_REDUCE_SUM, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceSum
  OP_PAD, //https://github.com/dmlc/tvm/blob/master/topi/python/topi/nn/pad.py
  OP_SHAPE, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Shape
  OP_SIZE, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Size
  OP_TOPK, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#TopK
  OP_WHERE, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Where
  OP_CEIL, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Ceil
  OP_CAST, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Cast
  OP_EXP, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Exp
  OP_ROUND, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Round
  OP_LOG, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Log
  OP_LOGICAL_NOT, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Not
  OP_SQRT, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sqrt
  OP_LEAKYRELU,
  OP_SLICE, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Slice
  OP_RESIZE, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Resize
  OP_PRELU, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#PRelu
};

#endif // _FLEXFLOW_CONST_H_
