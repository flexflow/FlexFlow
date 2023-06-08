#ifndef _FLEXFLOW_AGGREGATE_PARAMS_H
#define _FLEXFLOW_AGGREGATE_PARAMS_H

#include "core.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

#define AGGREGATE_MAX_K 4
#define AGGREGATE_MAX_BATCH_SIZE 64
#define AGGREGATE_MAX_N 12

namespace FlexFlow {

struct AggregateAttrs : public use_visitable_cmp<AggregateAttrs> {
  AggregateAttrs() = delete;
  AggregateAttrs(int n, float lambda_bal);

public:
  int n;
  float lambda_bal;
};

DataType get_datatype(AggregateAttrs const &);
bool is_valid(AggregateAttrs const &, ParallelTensorShape const &gate_preds,
              ParallelTensorShape const &gate_assign,
              ParallelTensorShape const &true_gate_assign,
              ParallelTensorShape const &full_gate_gradients,
              std::vector<ParallelTensorShape> const &exp_preds);
ParallelTensorShape
get_output_shape(AggregateAttrs const &, ParallelTensorShape const &gate_preds,
                 ParallelTensorShape const &gate_asign,
                 ParallelTensorShape const &true_gate_assign,
                 ParallelTensorShape const &full_gate_gradients,
                 std::vector<ParallelTensorShape> const &exp_preds);

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::AggregateAttrs, n, lambda_bal);
MAKE_VISIT_HASHABLE(::FlexFlow::AggregateAttrs);

namespace FlexFlow {

static_assert(is_valid_opattr<AggregateAttrs>::value,
              "AggregateAttrs must be a valid opattr (see core.h)");

}

#endif
