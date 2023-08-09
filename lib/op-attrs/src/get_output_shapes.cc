#include "op-attrs/get_output_shapes.h"

namespace FlexFlow {

ParallelTensorShape as_parallel(TensorShape const &);
std::vector<ParallelTensorShape> as_parallel(std::vector<TensorShape> const &);

TensorShape get_output_shape(AggregateAttrs const &attrs,
                             TensorShape const &gate_preds,
                             TensorShape const &gate_assign,
                             TensorShape const &true_gate_assign,
                             TensorShape const &full_gate_gradients,
                             std::vector<TensorShape> const &exp_preds) {
  return get_tensor_shape_unsafe(
      get_output_shape(attrs,
                       as_parallel(gate_preds),
                       as_parallel(gate_assign),
                       as_parallel(true_gate_assign),
                       as_parallel(full_gate_gradients),
                       as_parallel(exp_preds)));
}

} // namespace FlexFlow
