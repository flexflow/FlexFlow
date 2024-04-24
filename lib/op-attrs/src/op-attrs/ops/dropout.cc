#include "op-attrs/ops/dropout.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(DropoutAttrs const &, ParallelTensorShape const &) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
