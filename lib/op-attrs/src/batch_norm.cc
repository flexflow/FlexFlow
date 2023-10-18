#include "op-attrs/ops/batch_norm.h"
#include "utils/exception.h"

namespace FlexFlow {
// input: [b, c, h, w]
// output: [b, c, h, w]
ParallelTensorShape get_output_shape(BatchNormAttrs const &attrs,
                                     ParallelTensorShape const &input_shape) {
  if (!input.is_valid() || input.num_dims() != 4) {
    throw mk_runtime_error(
        "BatchNormAttrs::get_output_shape: input is invalid");
  }

  // the degree of the output is the same as the input
  return input_shape;
}

} // namespace FlexFlow
