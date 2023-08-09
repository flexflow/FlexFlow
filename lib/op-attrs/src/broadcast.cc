#include "op-attrs/ops/broadcast.h"

namespace FlexFlow {

BroadcastAttrs::BroadcastAttrs(
    stack_vector<int, MAX_TENSOR_DIM> const &target_dims)
    : target_dims(target_dims) {}

} // namespace FlexFlow
