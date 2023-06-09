#include "op-attrs/ops/replicate.h"

namespace FlexFlow {

ReplicateAttrs::ReplicateAttrs(ff_dim_t _dim, int _degree)
    : replicate_dim(_dim), replicate_degree(_degree) {}

} // namespace FlexFlow
