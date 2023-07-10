#include "op-attrs/ops/transpose.h"

namespace FlexFlow {

TransposeAttrs::TransposeAttrs(
    stack_vector<ff_dim_t, MAX_TENSOR_DIM> const &_perm)
    : perm(_perm) {}

} // namespace FlexFlow
