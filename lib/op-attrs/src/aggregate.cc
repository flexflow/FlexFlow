#include "op-attrs/ops/aggregate.h"
#include "utils/containers.h"

namespace FlexFlow {

DataType get_datatype(AggregateAttrs const &) {
  return DataType::FLOAT;
}

ParallelTensorShape
    get_output_shape(AggregateAttrs const &attrs,
                     ParallelTensorShape const &gate_preds,
                     ParallelTensorShape const &gate_assign,
                     ParallelTensorShape const &true_gate_assign,
                     ParallelTensorShape const &full_gate_gradients,
                     std::vector<ParallelTensorShape> const &exp_preds) {
  ParallelTensorShape output_shape = exp_preds.at(0);
  output_shape.data_type = DataType::FLOAT;
  ff_dim_t idx = ff_dim_t(gate_preds.dims.num_dims() - 1);
  output_shape.dims.at(idx) = gate_preds.dims.at(idx);
  return output_shape;
}

bool is_valid(AggregateAttrs const &attrs,
              ParallelTensorShape const &gate_preds,
              ParallelTensorShape const &gate_assign,
              ParallelTensorShape const &true_gate_assign,
              ParallelTensorShape const &full_gate_gradients,
              std::vector<ParallelTensorShape> const &exp_preds) {
  // FIXME: For now, set upper limits Better: Do as follows, but memory is
  // assigned per block, so requires to check that
  // https://stackoverflow.com/questions/5531247/allocating-shared-memory/5531640#5531640
  return attrs.n > 0 && attrs.n <= AGGREGATE_MAX_N &&
         gate_preds.at(ff_dim_t(0)).size <= AGGREGATE_MAX_K &&
         gate_preds.at(ff_dim_t(1)).size <= AGGREGATE_MAX_BATCH_SIZE &&
         attrs.n == exp_preds.size() && gate_preds.num_dims() == 3 &&
         gate_assign.num_dims() == 3 && true_gate_assign.num_dims() == 3 &&
         full_gate_gradients.num_dims() == 3 &&
         gate_preds.dims == gate_assign.dims &&
         gate_preds.dims == true_gate_assign.dims &&
         gate_preds.at(ff_dim_t(1)) == full_gate_gradients.at(ff_dim_t(1)) &&
         full_gate_gradients.at(ff_dim_t(0)).size == attrs.n &&
         are_all_same(exp_preds);
}

} // namespace FlexFlow
