#include "op-attrs/ops/repartition.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(RepartitionAttrs const &, ParallelTensorShape const &input_shape) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
