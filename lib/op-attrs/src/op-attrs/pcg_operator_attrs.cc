#include "op-attrs/pcg_operator_attrs.h"
#include "op-attrs/get_op_type.h"

namespace FlexFlow {

bool is_parallel_op(PCGOperatorAttrs const &attrs) {
  return (attrs.has<CombineAttrs>() || attrs.has<ReductionAttrs>() ||
          attrs.has<RepartitionAttrs>() || attrs.has<ReplicateAttrs>());
}

OperatorType get_op_type(PCGOperatorAttrs const &attrs) {
  return attrs.visit<OperatorType>(
      [](auto const &x) { return get_op_type(x); });
}

ComputationGraphOpAttrs
    compgraph_op_attrs_from_pcg_op_attrs(PCGOperatorAttrs const &op) {
  if (op.has<BatchMatmulAttrs>()) {
    return ComputationGraphOpAttrs{op.get<BatchMatmulAttrs>()};
  } else if (op.has<BatchNormAttrs>()) {
    return ComputationGraphOpAttrs{op.get<BatchNormAttrs>()};
  } else if (op.has<CastAttrs>()) {
    return ComputationGraphOpAttrs{op.get<CastAttrs>()};
  } else if (op.has<ConcatAttrs>()) {
    return ComputationGraphOpAttrs{op.get<ConcatAttrs>()};
  } else if (op.has<Conv2DAttrs>()) {
    return ComputationGraphOpAttrs{op.get<Conv2DAttrs>()};
  } else if (op.has<DropoutAttrs>()) {
    return ComputationGraphOpAttrs{op.get<DropoutAttrs>()};
  } else if (op.has<ElementBinaryAttrs>()) {
    return ComputationGraphOpAttrs{op.get<ElementBinaryAttrs>()};
  } else if (op.has<ElementUnaryAttrs>()) {
    return ComputationGraphOpAttrs{op.get<ElementUnaryAttrs>()};
  } else if (op.has<EmbeddingAttrs>()) {
    return ComputationGraphOpAttrs{op.get<EmbeddingAttrs>()};
  } else if (op.has<FlatAttrs>()) {
    return ComputationGraphOpAttrs{op.get<FlatAttrs>()};
  } else if (op.has<GatherAttrs>()) {
    return ComputationGraphOpAttrs{op.get<GatherAttrs>()};
  } else if (op.has<InputAttrs>()) {
    return ComputationGraphOpAttrs{op.get<InputAttrs>()};
  } else if (op.has<LayerNormAttrs>()) {
    return ComputationGraphOpAttrs{op.get<LayerNormAttrs>()};
  } else if (op.has<LinearAttrs>()) {
    return ComputationGraphOpAttrs{op.get<LinearAttrs>()};
  } else if (op.has<MultiHeadAttentionAttrs>()) {
    return ComputationGraphOpAttrs{op.get<MultiHeadAttentionAttrs>()};
  } else if (op.has<NoopAttrs>()) {
    return ComputationGraphOpAttrs{op.get<NoopAttrs>()};
  } else if (op.has<Pool2DAttrs>()) {
    return ComputationGraphOpAttrs{op.get<Pool2DAttrs>()};
  } else if (op.has<ReduceAttrs>()) {
    return ComputationGraphOpAttrs{op.get<ReduceAttrs>()};
  } else if (op.has<ReverseAttrs>()) {
    return ComputationGraphOpAttrs{op.get<ReverseAttrs>()};
  } else if (op.has<ReshapeAttrs>()) {
    return ComputationGraphOpAttrs{op.get<ReshapeAttrs>()};
  } else if (op.has<SplitAttrs>()) {
    return ComputationGraphOpAttrs{op.get<SplitAttrs>()};
  } else if (op.has<SoftmaxAttrs>()) {
    return ComputationGraphOpAttrs{op.get<SoftmaxAttrs>()};
  } else if (op.has<TopKAttrs>()) {
    return ComputationGraphOpAttrs{op.get<TopKAttrs>()};
  } else if (op.has<TransposeAttrs>()) {
    return ComputationGraphOpAttrs{op.get<TransposeAttrs>()};
  } else {
    throw mk_runtime_error(fmt::format(
        "Cannot convert parallel op to non-parallel, received {}", op));
  }
}

} // namespace FlexFlow
