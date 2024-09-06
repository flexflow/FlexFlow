#include "op-attrs/ops/split.h"

namespace FlexFlow {

std::vector<TensorShape> get_output_shapes(SplitAttrs const &,
                                           TensorShape const &) {
  NOT_IMPLEMENTED();
}

std::vector<ParallelTensorShape>
    get_output_shapes(SplitAttrs const &attrs,
                      ParallelTensorShape const &input_shape) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
