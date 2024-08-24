#include "op-attrs/pcg_operator_attrs.h"
#include "op-attrs/get_op_type.h"
#include "utils/overload.h"
#include "op-attrs/ops/linear.h"

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
  return op.visit<ComputationGraphOpAttrs>(overload{
      [](BatchMatmulAttrs const &attrs) {
        return ComputationGraphOpAttrs{attrs};
      },
      [](BatchNormAttrs const &attrs) {
        return ComputationGraphOpAttrs{attrs};
      },
      [](CastAttrs const &attrs) { return ComputationGraphOpAttrs{attrs}; },
      [](ConcatAttrs const &attrs) { return ComputationGraphOpAttrs{attrs}; },
      [](Conv2DAttrs const &attrs) { return ComputationGraphOpAttrs{attrs}; },
      [](DropoutAttrs const &attrs) { return ComputationGraphOpAttrs{attrs}; },
      [](ElementBinaryAttrs const &attrs) {
        return ComputationGraphOpAttrs{attrs};
      },
      [](ElementUnaryAttrs const &attrs) {
        return ComputationGraphOpAttrs{attrs};
      },
      [](EmbeddingAttrs const &attrs) {
        return ComputationGraphOpAttrs{attrs};
      },
      [](FlatAttrs const &attrs) { return ComputationGraphOpAttrs{attrs}; },
      [](GatherAttrs const &attrs) { return ComputationGraphOpAttrs{attrs}; },
      [](InputAttrs const &attrs) { return ComputationGraphOpAttrs{attrs}; },
      [](LayerNormAttrs const &attrs) {
        return ComputationGraphOpAttrs{attrs};
      },
      [](LinearAttrs const &attrs) { return ComputationGraphOpAttrs{attrs}; },
      [](MultiHeadAttentionAttrs const &attrs) {
        return ComputationGraphOpAttrs{attrs};
      },
      [](NoopAttrs const &attrs) { return ComputationGraphOpAttrs{attrs}; },
      [](Pool2DAttrs const &attrs) { return ComputationGraphOpAttrs{attrs}; },
      [](ReduceAttrs const &attrs) { return ComputationGraphOpAttrs{attrs}; },
      [](ReverseAttrs const &attrs) { return ComputationGraphOpAttrs{attrs}; },
      [](ReshapeAttrs const &attrs) { return ComputationGraphOpAttrs{attrs}; },
      [](SplitAttrs const &attrs) { return ComputationGraphOpAttrs{attrs}; },
      [](SoftmaxAttrs const &attrs) { return ComputationGraphOpAttrs{attrs}; },
      [](TopKAttrs const &attrs) { return ComputationGraphOpAttrs{attrs}; },
      [](TransposeAttrs const &attrs) {
        return ComputationGraphOpAttrs{attrs};
      },
      [](auto const &attrs) -> ComputationGraphOpAttrs {
        throw mk_runtime_error(fmt::format(
            "Cannot convert parallel op to non-parallel, received {}", attrs));
      },
  });
}

RecordFormatter as_dot(PCGOperatorAttrs const &attrs) {
  return attrs.visit<RecordFormatter>(overload {
    [](LinearAttrs const &l) { return as_dot(l); },
    [&](auto const &) { 
      RecordFormatter r;
      r << fmt::to_string(get_op_type(attrs));
      return r;
    },
  });
}


} // namespace FlexFlow
