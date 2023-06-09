#include "op-attrs/ops/layer_norm.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

LayerNormAttrs::LayerNormAttrs(
    stack_vector<ff_dim_t, MAX_TENSOR_DIM> const &_axes,
    bool _elementwise_affine,
    float _eps)
    : axes(_axes), elementwise_affine(_elementwise_affine), eps(_eps) {}

} // namespace FlexFlow
