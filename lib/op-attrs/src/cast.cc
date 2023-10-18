#include "op-attrs/ops/cast.h"
#include "utils/exception.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(CastAttrs const &attrs,
                                     ParallelTensorShape const &input_shape) {
  if (!input.is_valid()) {
    throw mk_runtime_error("CastAttrs::get_output_shape: input is invalid");
  }
  ParallelTensorShape output_shape = input_shape;
  output_shape.data_type = attrs.dtype;
  return output_shape;
}

} // namespace FlexFlow
