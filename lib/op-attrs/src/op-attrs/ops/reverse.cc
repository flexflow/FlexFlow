#include "op-attrs/ops/reverse.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

TensorShape reverse_get_output_shape(ReverseAttrs const &,
                                     TensorShape const &) {
  NOT_IMPLEMENTED();
}

ParallelTensorShape
    reverse_get_output_parallel_shape(ReverseAttrs const &attrs,
                                      ParallelTensorShape const &input_shape) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
