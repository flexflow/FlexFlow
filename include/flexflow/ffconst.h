#ifndef _FLEXFLOW_CONST_H_
#define _FLEXFLOW_CONST_H_

enum ActiMode {
  AC_MODE_NONE = 10,
  AC_MODE_RELU = 11,
  AC_MODE_SIGMOID = 12,
  AC_MODE_TANH = 13,
  AC_MODE_GELU = 14,
};

enum RegularizerMode {
  REG_MODE_NONE = 17,
  REG_MODE_L1 = 18,
  REG_MODE_L2 = 19,
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
  DT_BOOLEAN = 40,
  DT_INT32 = 41,
  DT_INT64 = 42,
  DT_HALF = 43,
  DT_FLOAT = 44,
  DT_DOUBLE = 45,
  DT_INT4 = 46,
  DT_INT8 = 47,
  DT_NONE = 49,
};

enum LossType {
  LOSS_CATEGORICAL_CROSSENTROPY = 50,
  LOSS_SPARSE_CATEGORICAL_CROSSENTROPY = 51,
  LOSS_MEAN_SQUARED_ERROR_AVG_REDUCE = 52,
  LOSS_MEAN_SQUARED_ERROR_SUM_REDUCE = 53,
  LOSS_IDENTITY = 54,
};

enum CompMode {
  COMP_MODE_TRAINING = 70,
  COMP_MODE_INFERENCE = 71,
};

enum ParameterSyncType {
  NONE = 80,
  PS = 81,
  NCCL = 82,
};

enum MetricsType {
  METRICS_ACCURACY = 1001,
  METRICS_CATEGORICAL_CROSSENTROPY = 1002,
  METRICS_SPARSE_CATEGORICAL_CROSSENTROPY = 1004,
  METRICS_MEAN_SQUARED_ERROR = 1008,
  METRICS_ROOT_MEAN_SQUARED_ERROR = 1016,
  METRICS_MEAN_ABSOLUTE_ERROR = 1032,
};

enum InferenceMode {
  INC_DECODING_MODE = 2001,
  BEAM_SEARCH_MODE = 2002,
  TREE_VERIFY_MODE = 2003,
};

// This is consistent with TASO's OpType
// https://github.com/jiazhihao/TASO/blob/master/include/taso/ops.h#L75-L138
enum OperatorType {
  OP_INPUT,
  OP_WEIGHT,
  OP_NOOP,
  OP_CONV2D,
  OP_DROPOUT,
  OP_LINEAR,
  OP_BATCHMATMUL,
  OP_POOL2D,
  OP_SCALAR_MULTIPLY,
  OP_SCALAR_ADD,
  OP_SCALAR_FLOOR_DIV,
  OP_SCALAR_TRUE_DIV,
  OP_SCALAR_SUB,
  OP_RELU,
  OP_IDENTITY,
  OP_SIGMOID,
  OP_TANH,
  OP_ELU,
  OP_FLAT,
  OP_SOFTMAX,
  OP_BATCHNORM,
  OP_CONCAT,
  OP_SPLIT,
  OP_EMBEDDING,
  OP_GROUP_BY,
  OP_CACHE,
  OP_AGGREGATE,
  OP_AGG_SPEC,
  // OP_ELEMENTWISE,
  OP_RESHAPE,
  OP_REVERSE,
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
  OP_SQUEEZE, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Squeeze
  OP_UNSQUEEZE, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Unsqueeze
  OP_EW_SUB,   // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sub
  OP_EW_DIV,   // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Div
  OP_EW_EQUAL, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Equal
  OP_EW_GREATER, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Greater
  OP_EW_LESS, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Less
  OP_EW_MAX,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Max
  OP_EW_MIN,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Min
  OP_REDUCE_ARGMAX, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#ArgMax
  OP_REDUCE_ARGMIN, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#ArgMin
  OP_REDUCE_MAX, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMax
  OP_REDUCE_MEAN, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMean
  OP_REDUCE_MIN, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMin
  OP_REDUCE_PROD, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceProd
  OP_REDUCE_SUM, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceSum
  OP_PAD, // https://github.com/dmlc/tvm/blob/master/topi/python/topi/nn/pad.py
  OP_SHAPE, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Shape
  OP_SIZE,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Size
  OP_TOPK,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#TopK
  OP_ARG_TOPK,
  OP_WHERE, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Where
  OP_CEIL,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Ceil
  OP_CAST,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Cast
  OP_EXP,   // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Exp
  OP_ROUND, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Round
  OP_LOG,   // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Log
  OP_LOGICAL_NOT, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Not
  OP_SQRT, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sqrt
  OP_SIN,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sin
  OP_COS,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Cos
  OP_LEAKYRELU,
  OP_SLICE,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Slice
  OP_RESIZE, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Resize
  OP_PRELU,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#PRelu
  OP_GELU,
  OP_MULTIHEAD_ATTENTION,
  OP_FUSED, // Fused operator type for internal fusion optimizations
  OP_RSQRT, // https://pytorch.org/docs/stable/generated/torch.rsqrt.html
  OP_POW,   // https://pytorch.org/docs/stable/generated/torch.pow.html
  OP_MEAN,  // https://pytorch.org/docs/stable/generated/torch.mean.html
  OP_LAYERNORM,
  OP_RESIDUAL_LAYERNORM,
  OP_ADD_BIAS_RESIDUAL_LAYERNORM,
  OP_SIGMOID_SILU_MULTI,
  OP_EXPERTS,
  OP_GATHER, // https://pytorch.org/docs/stable/generated/torch.gather.html
  OP_RMS_NORM,
  OP_RESIDUAL_RMS_NORM,
  OP_BEAM_TOPK,
  OP_ARGMAX,
  OP_INC_MULTIHEAD_SELF_ATTENTION,
  OP_SPEC_INC_MULTIHEAD_SELF_ATTENTION,
  OP_TREE_INC_MULTIHEAD_SELF_ATTENTION,
  OP_SAMPLING,
  // Parallel Ops
  OP_REPARTITION,
  OP_COMBINE,
  OP_REPLICATE,
  OP_REDUCTION,
  OP_PIPELINE,
  OP_ALLREDUCE,
  OP_FUSED_PARALLEL,
  OP_INVALID,
};

enum ModelType {
  UNKNOWN = 3001,
  LLAMA = 3002,
  OPT = 3003,
  FALCON = 3004,
  STARCODER = 3005,
  MPT = 3006
};

enum PMParameter {
  PM_OP_TYPE,            // AnyOp
  PM_NUM_INPUTS,         // AnyOp
  PM_NUM_OUTPUTS,        // AnyOp
  PM_GROUP,              // Conv2D
  PM_KERNEL_H,           // Conv2D, Pool2D
  PM_KERNEL_W,           // Conv2D, Pool2D
  PM_STRIDE_H,           // Conv2D, Pool2D
  PM_STRIDE_W,           // Conv2D, Pool2D
  PM_PADDING_H,          // Conv2D, Pool2D
  PM_PADDING_W,          // Conv2D, Pool2D
  PM_ACTI,               // Conv2D, Pool2D
  PM_NUMDIM,             // Concat, Transpose
  PM_AXIS,               // Concat, Split
  PM_PERM,               // Transpose
  PM_OUTSHUFFLE,         // Transpose
  PM_MERGE_GCONV_COUNT,  // MergeGConv
  PM_AXES,               // Squeeze, Unsqueeze, Reduce*
  PM_KEEP_DIMS,          // Reduce*
  PM_EPSILON,            // BatchNorm
  PM_REPARTITION_DIM,    // Repartition
  PM_REPARTITION_DEGREE, // Repartition
  PM_REPLICATE_DIM,      // Replicate
  PM_REPLICATE_DEGREE,   // Replicate
  PM_COMBINE_DIM,        // Combine
  PM_COMBINE_DEGREE,     // Combine
  PM_REDUCTION_DIM,      // Reduction
  PM_REDUCTION_DEGREE,   // Reduction
  PM_ALLREDUCE_DIM,      // AllReduce
  PM_SOFTMAX_DIM,        // Softmax
  PM_NUM_HEADS,          // MultiHeadAttention
  PM_INVALID,
  PM_PARALLEL_DIM,
  PM_PARALLEL_DEGREE,
  PM_PAD,
};

enum TNParameter {
  INPUT_0 = 100,
  INPUT_1 = 101,
  INPUT_2 = 102,
  INPUT_3 = 103,
  INPUT_4 = 104,
  INPUT_5 = 105,
  WEIGHT_0 = 200,
  WEIGHT_1 = 201,
  WEIGHT_2 = 202,
  WEIGHT_3 = 203,
  WEIGHT_4 = 204,
  WEIGHT_5 = 205,
  OUTPUT_0 = 300,
  OUTPUT_1 = 301,
  OUTPUT_2 = 302,
  OUTPUT_3 = 303,
  OUTPUT_4 = 304,
  OUTPUT_5 = 305,
};

enum DIMParameter {
  DIM_0 = 500,
  DIM_1 = 501,
  DIM_2 = 502,
  DIM_3 = 503,
  DIM_4 = 504,
  DIM_ND = 510,
};

enum {
  LAYER_GUID_FIRST_VALID = 1000000,
  LAYER_GUID_LAST_VALID = 1999999,
  OP_GUID_FIRST_VALID = 2000000,
  OP_GUID_LAST_VALID = 2999999,
  TENSOR_GUID_FIRST_VALID = 3000000,
  TENSOR_GUID_LAST_VALID = 3999999,
  PARALLEL_TENSOR_GUID_FIRST_VALID = 4000000,
  NODE_GUID_FIRST_VALID = 5000000,
};
#endif // _FLEXFLOW_CONST_H_
