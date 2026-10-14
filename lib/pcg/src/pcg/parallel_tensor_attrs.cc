#include "pcg/parallel_tensor_attrs.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/parallel_computation_graph/parallel_tensor_attrs.dtg.h"

namespace FlexFlow {

TensorAttrs get_piece_attrs(ParallelTensorAttrs const &parallel_attrs) {
  return TensorAttrs{get_piece_shape(parallel_attrs.shape),
                     parallel_attrs.create_grad};
}

ParallelTensorAttrs
    parallel_tensor_attrs_from_tensor_attrs(TensorAttrs const &tensor_attrs) {
  return ParallelTensorAttrs{
      lift_shape_to_parallel(tensor_attrs.shape),
      tensor_attrs.create_grad,
  };
}

} // namespace FlexFlow
