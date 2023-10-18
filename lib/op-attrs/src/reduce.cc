#include "op-attrs/ops/reduce.h"
#include "utils/exceptions.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(ReduceAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  NOT_IMPLEMENTED()
  // reduce is sum/max/min/mean, I think we just return 1D tensor
  // NOTE: how to implement this
}

} // namespace FlexFlow
