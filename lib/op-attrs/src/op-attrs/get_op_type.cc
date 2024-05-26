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
  return attrs.type;
}
OperatorType get_op_type(ElementUnaryAttrs const &attrs) {
  return attrs.op_type;
}
OperatorType get_op_type(ElementScalarUnaryAttrs const &attrs) {
  return attrs.op_type;
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
  return attrs.op_type;
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
