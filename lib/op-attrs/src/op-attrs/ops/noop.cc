#include "op-attrs/ops/noop.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(NoopAttrs const &, ParallelTensorShape const &input_shape) {
  return input_shape; 
}

} // namespace FlexFlow
