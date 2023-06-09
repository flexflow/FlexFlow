#include "op-attrs/get_op_type.h"

namespace FlexFlow {

OperatorType get_op_type(AggregateAttrs const &) {
  return Op::AGGREGATE;
}
OperatorType get_op_type(AggregateSpecAttrs const &) {
  return Op::AGG_SPEC;
}
OperatorType get_op_type(BatchMatmulAttrs const &) {
  return Op::BATCHMATMUL;
}
OperatorType get_op_type(BatchNormAttrs const &) {
  return Op::BATCHNORM;
}
OperatorType get_op_type(BroadcastAttrs const &) {
  return Op::BROADCAST;
}
OperatorType get_op_type(CastAttrs const &) {
  return Op::CAST;
}
OperatorType get_op_type(ConcatAttrs const &) {
  return Op::CONCAT;
}
OperatorType get_op_type(Conv2DAttrs const &) {
  return Op::CONV2D;
}
OperatorType get_op_type(DropoutAttrs const &) {
  return Op::DROPOUT;
}
OperatorType get_op_type(ElementBinaryAttrs const &attrs) {
  return attrs.type;
}
OperatorType get_op_type(ElementUnaryAttrs const &attrs) {
  return attrs.op;
}
OperatorType get_op_type(EmbeddingAttrs const &) {
  return Op::EMBEDDING;
}
OperatorType get_op_type(FlatAttrs const &) {
  return Op::FLAT;
}
OperatorType get_op_type(GatherAttrs const &) {
  return Op::GATHER;
}
OperatorType get_op_type(Group_byAttrs const &) {
  return Op::GROUP_BY;
}
OperatorType get_op_type(InputAttrs const &) {
  return Op::INPUT;
}
OperatorType get_op_type(LayerNormAttrs const &) {
  return Op::LAYERNORM;
}
OperatorType get_op_type(LinearAttrs const &) {
  return Op::LINEAR;
}
OperatorType get_op_type(MultiHeadAttentionAttrs const &) {
  return Op::MULTIHEAD_ATTENTION;
}
OperatorType get_op_type(NoopAttrs const &) {
  return Op::NOOP;
}
OperatorType get_op_type(Pool2DAttrs const &) {
  return Op::POOL2D;
}
OperatorType get_op_type(ReduceAttrs const &) {
  return Op::REDUCE_SUM;
}
OperatorType get_op_type(ReshapeAttrs const &) {
  return Op::RESHAPE;
}
OperatorType get_op_type(SplitAttrs const &) {
  return Op::SPLIT;
}
OperatorType get_op_type(SoftmaxAttrs const &) {
  return Op::SOFTMAX;
}
OperatorType get_op_type(TopKAttrs const &) {
  return Op::TOPK;
}
OperatorType get_op_type(TransposeAttrs const &) {
  return Op::TRANSPOSE;
}
OperatorType get_op_type(CombineAttrs const &) {
  return Op::COMBINE;
}
OperatorType get_op_type(ReductionAttrs const &) {
  return Op::REDUCTION;
}
OperatorType get_op_type(RepartitionAttrs const &) {
  return Op::REPARTITION;
}
OperatorType get_op_type(ReplicateAttrs const &) {
  return Op::REPLICATE;
}

} // namespace FlexFlow
