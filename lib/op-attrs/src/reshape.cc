#include "op-attrs/ops/reshape.h"

namespace FlexFlow {

ReshapeAttrs::ReshapeAttrs(TensorShape const &_shape) : shape(_shape) {}

} // namespace FlexFlow
