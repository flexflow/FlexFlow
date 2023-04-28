#ifndef _FLEXFLOW_FFCONST_UTILS_H
#define _FLEXFLOW_FFCONST_UTILS_H

#include "op-attrs/op-attrs.h"
#include <string>
#include "utils/fmt.h"

namespace FlexFlow {

std::string get_operator_type_name(OperatorType type);
bool is_parallel_op(OperatorType const &);

std::ostream &operator<<(std::ostream &, OperatorType);

}

namespace fmt {

template <>
struct formatter<::DataType> : formatter<string_view> {
  template <typename FormatContext>
  auto format(::DataType dt, FormatContext &ctx) -> decltype(ctx.out()) {
    string_view name = "unknown";
    switch (dt) {
      case DT_BOOLEAN: 
        name = "DT_BOOLEAN"; 
        break;
      case DT_INT32: 
        name = "DT_INT32";
        break;
      case DT_INT64:
        name = "DT_INT64";
        break;
      case DT_HALF:
        name = "DT_HALF";
        break;
      case DT_FLOAT:
        name = "DT_FLOAT";
        break;
      case DT_DOUBLE:
        name = "DT_DOUBLE";
        break;
      case DT_NONE:
        name = "DT_NONE";
        break;
    }
    return formatter<string_view>::format(name, ctx);
  }
};

template <>
struct formatter<::OperatorType> : formatter<string_view> {
  template <typename FormatContext>
  auto format(::OperatorType ot, FormatContext &ctx) -> decltype(ctx.out()) {
    string_view name = "unknown";
    switch (ot) {
      case OP_CONV2D:
        name = "Conv2D";
      case OP_DROPOUT:
        name = "Dropout";
      case OP_LINEAR:
        name = "Dense";
      case OP_BATCHMATMUL:
        name = "BatchMatMul";
      case OP_POOL2D:
        name = "Pool2D";
      case OP_SCALAR_MULTIPLY:
        name = "ScalarMultiply";
      case OP_SCALAR_ADD:
        name = "ScalarAdd";
      case OP_SCALAR_FLOOR_DIV:
        name = "ScalarFloorDiv";
      case OP_SCALAR_TRUE_DIV:
        name = "ScalarTrueDiv";
      case OP_SCALAR_SUB:
        name = "ScalarSub";
      case OP_RELU:
        name = "ReLU";
      case OP_SIGMOID:
        name = "Sigmoid";
      case OP_TANH:
        name = "Tanh";
      case OP_ELU:
        name = "Elu";
      case OP_FLAT:
        name = "Flat";
      case OP_SOFTMAX:
        name = "Softmax";
      case OP_BATCHNORM:
        name = "BatchNorm";
      case OP_CONCAT:
        name = "Concat";
      case OP_SPLIT:
        name = "Split";
      case OP_EMBEDDING:
        name = "Embedding";
      case OP_GATHER:
        name = "Gather";
      case OP_GROUP_BY:
        name = "Group_by";
      case OP_CACHE:
        name = "Cache";
      case OP_AGGREGATE:
        name = "Aggregate cooperation";
      case OP_AGG_SPEC:
        name = "Aggregate specification";
      case OP_RESHAPE:
        name = "Reshape";
      case OP_REVERSE:
        name = "Reverse";
      case OP_TRANSPOSE:
        name = "Transpose";
      case OP_EW_ADD:
        name = "Add";
      case OP_EW_MUL:
        name = "Mul";
      case OP_MATMUL:
        name = "Matmul";
      case OP_MUL:
        name = "Mul";
      case OP_ENLARGE:
        name = "Enlarge";
      case OP_SQUEEZE:
        name = "Squeeze";
      case OP_UNSQUEEZE:
        name = "Unsqueeze";
      case OP_EW_SUB:
        name = "Sub";
      case OP_EW_DIV:
        name = "Div";
      case OP_EW_EQUAL:
        name = "Equal";
      case OP_EW_GREATER:
        name = "Greater";
      case OP_EW_LESS:
        name = "Less";
      case OP_EW_MAX:
        name = "Max";
      case OP_EW_MIN:
        name = "Min";
      case OP_REDUCE_ARGMAX:
        name = "ReduceArgMax";
      case OP_REDUCE_ARGMIN:
        name = "ReduceArgMin";
      case OP_REDUCE_MAX:
        name = "ReduceMax";
      case OP_REDUCE_MEAN:
        name = "ReduceMean";
      case OP_REDUCE_MIN:
        name = "ReduceMin";
      case OP_REDUCE_PROD:
        name = "ReduceProd";
      case OP_REDUCE_SUM:
        name = "ReduceSum";
      case OP_PAD:
        name = "Pad";
      case OP_SHAPE:
        name = "Shape";
      case OP_SIZE:
        name = "Size";
      case OP_TOPK:
        name = "TopK";
      case OP_WHERE:
        name = "Where";
      case OP_CEIL:
        name = "Ceil";
      case OP_CAST:
        name = "Cast";
      case OP_EXP:
        name = "Exp";
      case OP_SIN:
        name = "Sin";
      case OP_COS:
        name = "Cos";
      case OP_ROUND:
        name = "Round";
      case OP_LOG:
        name = "Log";
      case OP_LOGICAL_NOT:
        name = "LogicalNot";
      case OP_SQRT:
        name = "Sqrt";
      case OP_LEAKYRELU:
        name = "LeakyReLU";
      case OP_SLICE:
        name = "Slice";
      case OP_RESIZE:
        name = "Resize";
      case OP_PRELU:
        name = "PReLU";
      case OP_MULTIHEAD_ATTENTION:
        name = "MultiHeadAttention";
      case OP_INPUT:
        name = "Input";
      case OP_WEIGHT:
        name = "Weight";
      case OP_NOOP:
        name = "NoOp";
      case OP_FUSED:
        name = "FusedOp";
      case OP_RSQRT:
        name = "Rsqrt";
      case OP_POW:
        name = "Pow";
      case OP_MEAN:
        name = "Mean";
      case OP_LAYERNORM:
        name = "LayerNorm";
      case OP_IDENTITY:
        name = "Identity";
      // Parallel Ops
      case OP_REPARTITION:
        name = "Repartition";
      case OP_COMBINE:
        name = "Combine";
      case OP_REPLICATE:
        name = "Replicate";
      case OP_REDUCTION:
        name = "Reduction";
      case OP_PIPELINE:
        name = "Pipeline";
      case OP_FUSED_PARALLEL:
        name = "FusedParallelOp";
    }
    return formatter<string_view>::format(name, ctx);
  }
};

template <>
struct formatter<ParameterSyncType> : formatter<string_view> {
  template <typename FormatContext>
  auto format(ParameterSyncType d, FormatContext& ctx) const -> decltype(ctx.out()) {
    string_view name = "unknown";
    switch (d) {
      case NONE: name = "NONE"; break;
      case PS: name = "PS"; break;
      case NCCL: name = "NCCL"; break;
    }
    return formatter<string_view>::format(name, ctx);
  } 
  
};

template <>
struct formatter<LossType> : formatter<string_view> {
  template <typename FormatContext>
  auto format(LossType d, FormatContext& ctx) const -> decltype(ctx.out()) {
    string_view name = "unknown";
    switch (d) {
      case LOSS_CATEGORICAL_CROSSENTROPY: name = "CategoricalCrossEntropy"; break;
      case LOSS_SPARSE_CATEGORICAL_CROSSENTROPY: name = "SparseCategoricalCrossEntropy"; break;
      case LOSS_MEAN_SQUARED_ERROR_AVG_REDUCE: name = "MeanSquaredErrorAvgReduce"; break;
      case LOSS_MEAN_SQUARED_ERROR_SUM_REDUCE: name = "MeanSquaredErrorSumReduce"; break;
      case LOSS_IDENTITY: name = "Identity"; break;
    }
    return formatter<string_view>::format(name, ctx);
  } 
  
};

}

#endif
