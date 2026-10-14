#include "op-attrs/get_op_type.h"

namespace FlexFlow {

OperatorType get_op_type(BatchMatmulAttrs const &) {
  return OperatorType::BATCHMATMUL;
}

OperatorType get_op_type(BatchNormAttrs const &) {
  return OperatorType::BATCHNORM;
}

OperatorType get_op_type(BroadcastAttrs const &) {
  return OperatorType::BROADCAST;
}

OperatorType get_op_type(CastAttrs const &) {
  return OperatorType::CAST;
}

OperatorType get_op_type(ConcatAttrs const &) {
  return OperatorType::CONCAT;
}

OperatorType get_op_type(Conv2DAttrs const &) {
  return OperatorType::CONV2D;
}

OperatorType get_op_type(DropoutAttrs const &) {
  return OperatorType::DROPOUT;
}

OperatorType get_op_type(ElementBinaryAttrs const &attrs) {
  switch (attrs.op) {
    case ElementBinaryOp::ADD:
      return OperatorType::EW_ADD;
    case ElementBinaryOp::SUBTRACT:
      return OperatorType::EW_SUB;
    case ElementBinaryOp::MULTIPLY:
      return OperatorType::EW_MUL;
    case ElementBinaryOp::DIVIDE:
      return OperatorType::EW_DIV;
    case ElementBinaryOp::MAX:
      return OperatorType::EW_MAX;
    case ElementBinaryOp::MIN:
      return OperatorType::EW_MIN;
    default:
      PANIC("Unknown ElementBinaryOp {}", attrs.op);
  }
}

OperatorType get_op_type(ElementUnaryAttrs const &attrs) {
  switch (attrs.op) {
    case ElementUnaryOp::RELU:
      return OperatorType::RELU;
    case ElementUnaryOp::IDENTITY:
      return OperatorType::IDENTITY;
    case ElementUnaryOp::GELU:
      return OperatorType::GELU;
    case ElementUnaryOp::SIGMOID:
      return OperatorType::SIGMOID;
    case ElementUnaryOp::TANH:
      return OperatorType::TANH;
    case ElementUnaryOp::ELU:
      return OperatorType::ELU;
    case ElementUnaryOp::SILU:
      return OperatorType::SILU;
    case ElementUnaryOp::SIN:
      return OperatorType::SIN;
    case ElementUnaryOp::COS:
      return OperatorType::COS;
    case ElementUnaryOp::RSQRT:
      return OperatorType::RSQRT;
    case ElementUnaryOp::SCALAR_ADD:
      return OperatorType::SCALAR_ADD;
    case ElementUnaryOp::SCALAR_SUB:
      return OperatorType::SCALAR_SUB;
    case ElementUnaryOp::SCALAR_MULTIPLY:
      return OperatorType::SCALAR_MULTIPLY;
    case ElementUnaryOp::SCALAR_TRUE_DIV:
      return OperatorType::SCALAR_TRUE_DIV;
    case ElementUnaryOp::POW:
      return OperatorType::POW;
    case ElementUnaryOp::EXP:
      return OperatorType::EXP;
    default:
      PANIC("Unknown ElementUnaryOp {}", attrs.op);
  }
}

OperatorType get_op_type(EmbeddingAttrs const &) {
  return OperatorType::EMBEDDING;
}

OperatorType get_op_type(FlatAttrs const &) {
  return OperatorType::FLAT;
}

OperatorType get_op_type(GatherAttrs const &) {
  return OperatorType::GATHER;
}

OperatorType get_op_type(InputAttrs const &) {
  return OperatorType::INPUT;
}

OperatorType get_op_type(LayerNormAttrs const &) {
  return OperatorType::LAYERNORM;
}

OperatorType get_op_type(LinearAttrs const &) {
  return OperatorType::LINEAR;
}

OperatorType get_op_type(MultiHeadAttentionAttrs const &) {
  return OperatorType::MULTIHEAD_ATTENTION;
}

OperatorType get_op_type(NoopAttrs const &) {
  return OperatorType::NOOP;
}

OperatorType get_op_type(Pool2DAttrs const &) {
  return OperatorType::POOL2D;
}

OperatorType get_op_type(ReduceAttrs const &attrs) {
  switch (attrs.op) {
    case ReduceOp::SUM:
      return OperatorType::REDUCE_SUM;
    case ReduceOp::AVG:
      return OperatorType::REDUCE_MEAN;
    default:
      PANIC("Unknown ReduceOp {}", attrs.op);
  }
}

OperatorType get_op_type(ReshapeAttrs const &) {
  return OperatorType::RESHAPE;
}

OperatorType get_op_type(ReverseAttrs const &) {
  return OperatorType::REVERSE;
}

OperatorType get_op_type(SplitAttrs const &) {
  return OperatorType::SPLIT;
}

OperatorType get_op_type(SoftmaxAttrs const &) {
  return OperatorType::SOFTMAX;
}

OperatorType get_op_type(TopKAttrs const &) {
  return OperatorType::TOPK;
}

OperatorType get_op_type(TransposeAttrs const &) {
  return OperatorType::TRANSPOSE;
}

OperatorType get_op_type(UpsampleAttrs const &) {
  return OperatorType::UPSAMPLE;
}

OperatorType get_op_type(CombineAttrs const &) {
  return OperatorType::COMBINE;
}

OperatorType get_op_type(ReductionAttrs const &) {
  return OperatorType::REDUCTION;
}

OperatorType get_op_type(RepartitionAttrs const &) {
  return OperatorType::REPARTITION;
}

OperatorType get_op_type(ReplicateAttrs const &) {
  return OperatorType::REPLICATE;
}

OperatorType get_op_type(WeightAttrs const &) {
  return OperatorType::WEIGHT;
}

} // namespace FlexFlow
