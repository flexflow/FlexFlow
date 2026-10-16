#include "op-attrs/ops/noop.h"

namespace FlexFlow {

TensorShape noop_get_output_shape(NoopAttrs const &,
                                  TensorShape const &input_shape) {
  return input_shape;
}

ParallelTensorShape
    noop_get_output_parallel_shape(NoopAttrs const &,
                                   ParallelTensorShape const &input_shape) {
  return input_shape;
}

} // namespace FlexFlow
