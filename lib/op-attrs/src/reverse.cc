#include "op-attrs/ops/reverse.h"
#include "op-attrs/ff_dim.h"
#include "utils/exception.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(ReverseAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  if (attrs.axis < 0 || attrs.axis >= input.num_dims()) {
    throw mk_runtime_error("ReverseAttrs: axis is invalid");
  }
  ParallelTensorShape output = input;
  // output degree is same as input degree, because it's just reverse operation
  return output;
}

}; // namespace FlexFlow
