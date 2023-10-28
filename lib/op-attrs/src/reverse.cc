#include "op-attrs/ops/reverse.h"
#include "op-attrs/ff_dim.h"
#include "utils/exception.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(ReverseAttrs const &attrs,
                                     ParallelTensorShape const &input_shape) {
  if (attrs.axis < 0 || attrs.axis >= input_shape.num_dims()) {
    throw mk_runtime_error("ReverseAttrs: axis is invalid");
  }
  return input_shape;
}

}; // namespace FlexFlow
