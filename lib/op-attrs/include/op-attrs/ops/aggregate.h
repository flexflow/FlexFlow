#ifndef _FLEXFLOW_AGGREGATE_PARAMS_H
#define _FLEXFLOW_AGGREGATE_PARAMS_H

#include "core.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

#define AGGREGATE_MAX_K 4
#define AGGREGATE_MAX_BATCH_SIZE 64
#define AGGREGATE_MAX_N 12

namespace FlexFlow {

struct AggregateAttrs {
  req<int> n;
  req<float> lambda_bal;
};
FF_VISITABLE_STRUCT(AggregateAttrs, n, lambda_bal);

DataType get_datatype(AggregateAttrs const &);
bool is_valid(AggregateAttrs const &,
              ParallelTensorShape const &gate_preds,
              ParallelTensorShape const &gate_assign,
              ParallelTensorShape const &true_gate_assign,
              ParallelTensorShape const &full_gate_gradients,
              std::vector<ParallelTensorShape> const &exp_preds);
ParallelTensorShape
    get_output_shape(AggregateAttrs const &,
                     ParallelTensorShape const &gate_preds,
                     ParallelTensorShape const &gate_asign,
                     ParallelTensorShape const &true_gate_assign,
                     ParallelTensorShape const &full_gate_gradients,
                     std::vector<ParallelTensorShape> const &exp_preds);

CHECK_VALID_OP_ATTR(AggregateAttrs);
} // namespace FlexFlow

#endif
