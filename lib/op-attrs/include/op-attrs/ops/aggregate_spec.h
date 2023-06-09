#ifndef _FLEXFLOW_AGGREGATE_SPEC_ATTRS_H
#define _FLEXFLOW_AGGREGATE_SPEC_ATTRS_H

#include "core.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct AggregateSpecAttrs : public use_visitable_cmp<AggregateSpecAttrs> {
public:
  AggregateSpecAttrs() = delete;
  AggregateSpecAttrs(int n, float lambda_bal);

public:
  int n;
  float lambda_bal;
};

ParallelTensorShape
    get_output_shape(AggregateSpecAttrs const &,
                     ParallelTensorShape const &gate_preds,
                     ParallelTensorShape const &gate_assign,
                     ParallelTensorShape const &true_gate_assign,
                     ParallelTensorShape const &gate_gradients_full,
                     std::vector<ParallelTensorShape> const &exp_preds);

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::AggregateSpecAttrs, n, lambda_bal);
MAKE_VISIT_HASHABLE(::FlexFlow::AggregateSpecAttrs);

namespace FlexFlow {

static_assert(is_valid_opattr<AggregateSpecAttrs>::value,
              "AggregateSpecAttrs must be a valid opattr (see core.h)");

}

#endif
