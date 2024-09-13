#include "op-attrs/ops/input.h"

namespace FlexFlow {

TensorShape get_output_shape(InputAttrs const &) {
  NOT_IMPLEMENTED();
}

ParallelTensorShape get_output_parallel_tensor_shape(InputAttrs const &) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
