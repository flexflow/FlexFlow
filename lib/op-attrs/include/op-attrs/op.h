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
  CACHE,
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
        break;
      case Op::DROPOUT:
        name = "Dropout";
        break;
      case Op::LINEAR:
        name = "Dense";
        break;
      case Op::BATCHMATMUL:
        name = "BatchMatMul";
        break;
      case Op::POOL2D:
        name = "Pool2D";
        break;
      case Op::SCALAR_MULTIPLY:
        name = "ScalarMultiply";
        break;
      case Op::SCALAR_ADD:
        name = "ScalarAdd";
        break;
      case Op::SCALAR_FLOOR_DIV:
        name = "ScalarFloorDiv";
        break;
      case Op::SCALAR_TRUE_DIV:
        name = "ScalarTrueDiv";
        break;
      case Op::SCALAR_SUB:
        name = "ScalarSub";
        break;
      case Op::RELU:
        name = "ReLU";
        break;
      case Op::SIGMOID:
        name = "Sigmoid";
        break;
      case Op::TANH:
        name = "Tanh";
        break;
      case Op::ELU:
        name = "Elu";
        break;
      case Op::FLAT:
        name = "Flat";
        break;
      case Op::SOFTMAX:
        name = "Softmax";
        break;
      case Op::BATCHNORM:
        name = "BatchNorm";
        break;
      case Op::CONCAT:
        name = "Concat";
        break;
      case Op::SPLIT:
        name = "Split";
        break;
      case Op::EMBEDDING:
        name = "Embedding";
        break;
      case Op::GATHER:
        name = "Gather";
        break;
      case Op::CACHE:
        name = "Cache";
        break;
      case Op::RESHAPE:
        name = "Reshape";
        break;
      case Op::REVERSE:
        name = "Reverse";
        break;
      case Op::TRANSPOSE:
        name = "Transpose";
        break;
      case Op::EW_ADD:
        name = "Add";
        break;
      case Op::EW_MUL:
        name = "Mul";
        break;
      case Op::MATMUL:
        name = "Matmul";
        break;
      case Op::MUL:
        name = "Mul";
        break;
      case Op::ENLARGE:
        name = "Enlarge";
        break;
      case Op::SQUEEZE:
        name = "Squeeze";
        break;
      case Op::UNSQUEEZE:
        name = "Unsqueeze";
        break;
      case Op::EW_SUB:
        name = "Sub";
        break;
      case Op::EW_DIV:
        name = "Div";
        break;
      case Op::EW_EQUAL:
        name = "Equal";
        break;
      case Op::EW_GREATER:
        name = "Greater";
        break;
      case Op::EW_LESS:
        name = "Less";
        break;
      case Op::EW_MAX:
        name = "Max";
        break;
      case Op::EW_MIN:
        name = "Min";
        break;
      case Op::REDUCE_ARGMAX:
        name = "ReduceArgMax";
        break;
      case Op::REDUCE_ARGMIN:
        name = "ReduceArgMin";
        break;
      case Op::REDUCE_MAX:
        name = "ReduceMax";
        break;
      case Op::REDUCE_MEAN:
        name = "ReduceMean";
        break;
      case Op::REDUCE_MIN:
        name = "ReduceMin";
        break;
      case Op::REDUCE_PROD:
        name = "ReduceProd";
        break;
      case Op::REDUCE_SUM:
        name = "ReduceSum";
        break;
      case Op::PAD:
        name = "Pad";
        break;
      case Op::SHAPE:
        name = "Shape";
        break;
      case Op::SIZE:
        name = "Size";
        break;
      case Op::TOPK:
        name = "TopK";
        break;
      case Op::WHERE:
        name = "Where";
        break;
      case Op::CEIL:
        name = "Ceil";
        break;
      case Op::CAST:
        name = "Cast";
        break;
      case Op::EXP:
        name = "Exp";
        break;
      case Op::SIN:
        name = "Sin";
        break;
      case Op::COS:
        name = "Cos";
        break;
      case Op::ROUND:
        name = "Round";
        break;
      case Op::LOG:
        name = "Log";
        break;
      case Op::LOGICAL_NOT:
        name = "LogicalNot";
        break;
      case Op::SQRT:
        name = "Sqrt";
        break;
      case Op::LEAKYRELU:
        name = "LeakyReLU";
        break;
      case Op::SLICE:
        name = "Slice";
        break;
      case Op::RESIZE:
        name = "Resize";
        break;
      case Op::PRELU:
        name = "PReLU";
        break;
      case Op::MULTIHEAD_ATTENTION:
        name = "MultiHeadAttention";
        break;
      case Op::INPUT:
        name = "Input";
        break;
      case Op::WEIGHT:
        name = "Weight";
        break;
      case Op::NOOP:
        name = "NoOp";
        break;
      case Op::FUSED:
        name = "FusedOp";
        break;
      case Op::RSQRT:
        name = "Rsqrt";
        break;
      case Op::POW:
        name = "Pow";
        break;
      case Op::MEAN:
        name = "Mean";
        break;
      case Op::LAYERNORM:
        name = "LayerNorm";
        break;
      case Op::IDENTITY:
        name = "Identity";
        break;
      // Parallel Ops
      case Op::REPARTITION:
        name = "Repartition";
        break;
      case Op::COMBINE:
        name = "Combine";
        break;
      case Op::REPLICATE:
        name = "Replicate";
        break;
      case Op::REDUCTION:
        name = "Reduction";
        break;
      case Op::PIPELINE:
        name = "Pipeline";
        break;
      case Op::FUSED_PARALLEL:
        name = "FusedParallelOp";
        break;
      case Op::GELU:
        name = "GeLU";
        break;
      case Op::BROADCAST:
        name = "Broadcast";
        break;
      case Op::BATCH:
        name = "Batch";
        break;
    }
    return formatter<string_view>::format(name, ctx);
  }
};

} // namespace fmt

#endif
