#ifndef _FLEXFLOW_OPATTRS_INCLUDE_OPATTRS_OP_H
#define _FLEXFLOW_OPATTRS_INCLUDE_OPATTRS_OP_H

#include "utils/fmt.h"

namespace FlexFlow {

enum class Op {
  NOOP,
  INPUT,
  WEIGHT,
  CONV2D,
  DROPOUT,
  LINEAR,
  BATCHMATMUL,
  POOL2D,
  SCALAR_MULTIPLY,
  SCALAR_ADD,
  SCALAR_FLOOR_DIV,
  SCALAR_TRUE_DIV,
  SCALAR_SUB,
  RELU,
  IDENTITY,
  SIGMOID,
  TANH,
  ELU,
  FLAT,
  SOFTMAX,
  BATCHNORM,
  CONCAT,
  SPLIT,
  EMBEDDING,
  GROUP_BY,
  CACHE,
  AGGREGATE,
  AGG_SPEC,
  // OP_ELEMENTWISE,
  RESHAPE,
  REVERSE,
  TRANSPOSE,
  EW_ADD,
  EW_MUL,
  MATMUL,
  MUL,
  ENLARGE,
  SQUEEZE, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Squeeze
  UNSQUEEZE, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Unsqueeze
  EW_SUB,   // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sub
  EW_DIV,   // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Div
  EW_EQUAL, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Equal
  EW_GREATER, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Greater
  EW_LESS,    // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Less
  EW_MAX,     // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Max
  EW_MIN,     // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Min
  REDUCE_ARGMAX, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#ArgMax
  REDUCE_ARGMIN, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#ArgMin
  REDUCE_MAX, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMax
  REDUCE_MEAN, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMean
  REDUCE_MIN, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMin
  REDUCE_PROD, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceProd
  REDUCE_SUM, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceSum
  PAD,   // https://github.com/dmlc/tvm/blob/master/topi/python/topi/nn/pad.py
  SHAPE, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Shape
  SIZE,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Size
  TOPK,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#TopK
  WHERE, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Where
  CEIL,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Ceil
  CAST,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Cast
  EXP,   // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Exp
  ROUND, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Round
  LOG,   // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Log
  LOGICAL_NOT, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Not
  SQRT, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sqrt
  SIN,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sin
  COS,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Cos
  LEAKYRELU,
  SLICE,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Slice
  RESIZE, // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Resize
  PRELU,  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#PRelu
  GELU,
  MULTIHEAD_ATTENTION,
  FUSED, // Fused operator type for internal fusion optimizations
  RSQRT, // https://pytorch.org/docs/stable/generated/torch.rsqrt.html
  POW,   // https://pytorch.org/docs/stable/generated/torch.pow.html
  MEAN,  // https://pytorch.org/docs/stable/generated/torch.mean.html
  LAYERNORM,
  GATHER, // https://pytorch.org/docs/stable/generated/torch.gather.html
  BROADCAST,
  // Parallel Ops
  REPARTITION,
  COMBINE,
  REPLICATE,
  REDUCTION,
  BATCH,
  PIPELINE,
  FUSED_PARALLEL,
};

using OperatorType = Op;

std::string get_operator_type_name(Op op);

} // namespace FlexFlow

namespace fmt {

template <>
struct formatter<::FlexFlow::Op> : formatter<string_view> {
  template <typename FormatContext>
  auto format(::FlexFlow::Op ot, FormatContext &ctx) -> decltype(ctx.out()) {
    using namespace FlexFlow;

    string_view name = "unknown";
    switch (ot) {
      case Op::CONV2D:
        name = "Conv2D";
      case Op::DROPOUT:
        name = "Dropout";
      case Op::LINEAR:
        name = "Dense";
      case Op::BATCHMATMUL:
        name = "BatchMatMul";
      case Op::POOL2D:
        name = "Pool2D";
      case Op::SCALAR_MULTIPLY:
        name = "ScalarMultiply";
      case Op::SCALAR_ADD:
        name = "ScalarAdd";
      case Op::SCALAR_FLOOR_DIV:
        name = "ScalarFloorDiv";
      case Op::SCALAR_TRUE_DIV:
        name = "ScalarTrueDiv";
      case Op::SCALAR_SUB:
        name = "ScalarSub";
      case Op::RELU:
        name = "ReLU";
      case Op::SIGMOID:
        name = "Sigmoid";
      case Op::TANH:
        name = "Tanh";
      case Op::ELU:
        name = "Elu";
      case Op::FLAT:
        name = "Flat";
      case Op::SOFTMAX:
        name = "Softmax";
      case Op::BATCHNORM:
        name = "BatchNorm";
      case Op::CONCAT:
        name = "Concat";
      case Op::SPLIT:
        name = "Split";
      case Op::EMBEDDING:
        name = "Embedding";
      case Op::GATHER:
        name = "Gather";
      case Op::GROUP_BY:
        name = "Group_by";
      case Op::CACHE:
        name = "Cache";
      case Op::AGGREGATE:
        name = "Aggregate cooperation";
      case Op::AGG_SPEC:
        name = "Aggregate specification";
      case Op::RESHAPE:
        name = "Reshape";
      case Op::REVERSE:
        name = "Reverse";
      case Op::TRANSPOSE:
        name = "Transpose";
      case Op::EW_ADD:
        name = "Add";
      case Op::EW_MUL:
        name = "Mul";
      case Op::MATMUL:
        name = "Matmul";
      case Op::MUL:
        name = "Mul";
      case Op::ENLARGE:
        name = "Enlarge";
      case Op::SQUEEZE:
        name = "Squeeze";
      case Op::UNSQUEEZE:
        name = "Unsqueeze";
      case Op::EW_SUB:
        name = "Sub";
      case Op::EW_DIV:
        name = "Div";
      case Op::EW_EQUAL:
        name = "Equal";
      case Op::EW_GREATER:
        name = "Greater";
      case Op::EW_LESS:
        name = "Less";
      case Op::EW_MAX:
        name = "Max";
      case Op::EW_MIN:
        name = "Min";
      case Op::REDUCE_ARGMAX:
        name = "ReduceArgMax";
      case Op::REDUCE_ARGMIN:
        name = "ReduceArgMin";
      case Op::REDUCE_MAX:
        name = "ReduceMax";
      case Op::REDUCE_MEAN:
        name = "ReduceMean";
      case Op::REDUCE_MIN:
        name = "ReduceMin";
      case Op::REDUCE_PROD:
        name = "ReduceProd";
      case Op::REDUCE_SUM:
        name = "ReduceSum";
      case Op::PAD:
        name = "Pad";
      case Op::SHAPE:
        name = "Shape";
      case Op::SIZE:
        name = "Size";
      case Op::TOPK:
        name = "TopK";
      case Op::WHERE:
        name = "Where";
      case Op::CEIL:
        name = "Ceil";
      case Op::CAST:
        name = "Cast";
      case Op::EXP:
        name = "Exp";
      case Op::SIN:
        name = "Sin";
      case Op::COS:
        name = "Cos";
      case Op::ROUND:
        name = "Round";
      case Op::LOG:
        name = "Log";
      case Op::LOGICAL_NOT:
        name = "LogicalNot";
      case Op::SQRT:
        name = "Sqrt";
      case Op::LEAKYRELU:
        name = "LeakyReLU";
      case Op::SLICE:
        name = "Slice";
      case Op::RESIZE:
        name = "Resize";
      case Op::PRELU:
        name = "PReLU";
      case Op::MULTIHEAD_ATTENTION:
        name = "MultiHeadAttention";
      case Op::INPUT:
        name = "Input";
      case Op::WEIGHT:
        name = "Weight";
      case Op::NOOP:
        name = "NoOp";
      case Op::FUSED:
        name = "FusedOp";
      case Op::RSQRT:
        name = "Rsqrt";
      case Op::POW:
        name = "Pow";
      case Op::MEAN:
        name = "Mean";
      case Op::LAYERNORM:
        name = "LayerNorm";
      case Op::IDENTITY:
        name = "Identity";
      // Parallel Ops
      case Op::REPARTITION:
        name = "Repartition";
      case Op::COMBINE:
        name = "Combine";
      case Op::REPLICATE:
        name = "Replicate";
      case Op::REDUCTION:
        name = "Reduction";
      case Op::PIPELINE:
        name = "Pipeline";
      case Op::FUSED_PARALLEL:
        name = "FusedParallelOp";
      case Op::GELU:
        name = "GeLU";
      case Op::BROADCAST:
        name = "Broadcast";
      case Op::BATCH:
        name = "Batch";
    }
    return formatter<string_view>::format(name, ctx);
  }
};

} // namespace fmt

#endif
