#include "op-attrs/ops/batch_norm.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(BatchNormAttrs const &, ParallelTensorShape const &) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
