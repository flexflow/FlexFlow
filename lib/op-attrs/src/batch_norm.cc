#include "op-attrs/ops/batch_norm.h"
#include "utils/exception.h"

namespace FlexFlow {
// input_shape: [b, c, h, w]
// output: [b, c, h, w]
ParallelTensorShape get_output_shape(BatchNormAttrs const &attrs,
                                     ParallelTensorShape const &input_shape) {
  if (!input_shape.is_valid() || input_shape.num_dims() != 4) {
    throw mk_runtime_error(
        "BatchNormAttrs::get_output_shape: input_shape is invalid");
  }

  // the degree of the output is the same as the input_shape
  return input_shape;
}

} // namespace FlexFlow
