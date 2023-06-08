#include "op-attrs/ops/split.h"

namespace FlexFlow {

SplitAttrs::SplitAttrs(stack_vector<int, MAX_NUM_OUTPUTS> const &_splits,
                       ff_dim_t _axis)
    : splits(_splits), axis(_axis) {}

} // namespace FlexFlow
