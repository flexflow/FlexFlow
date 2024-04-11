#include "op-attrs/tensor_shape.h"

namespace FlexFlow {

size_t dim_at_idx(TensorShape const &s, ff_dim_t idx) {
  return dim_at_idx(s.dims, idx);
}

} // namespace FlexFlow
