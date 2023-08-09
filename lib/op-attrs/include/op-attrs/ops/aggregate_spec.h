#ifndef _FLEXFLOW_AGGREGATE_SPEC_ATTRS_H
#define _FLEXFLOW_AGGREGATE_SPEC_ATTRS_H

#include "core.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct AggregateSpecAttrs {
  req<int> n;
  req<float> lambda_bal;
};
FF_VISITABLE_STRUCT(AggregateSpecAttrs, n, lambda_bal);

ParallelTensorShape
    get_output_shape(AggregateSpecAttrs const &,
                     ParallelTensorShape const &gate_preds,
                     ParallelTensorShape const &gate_assign,
                     ParallelTensorShape const &true_gate_assign,
                     ParallelTensorShape const &gate_gradients_full,
                     std::vector<ParallelTensorShape> const &exp_preds);

CHECK_VALID_OP_ATTR(AggregateSpecAttrs);
} // namespace FlexFlow

#endif
