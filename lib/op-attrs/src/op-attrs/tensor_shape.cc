#include "op-attrs/tensor_shape.h"
#include "op-attrs/tensor_dims.h"

namespace FlexFlow {

size_t num_dims(TensorShape const &s) {
  return s.dims.ff_ordered.size();
}

size_t dim_at_idx(TensorShape const &s, ff_dim_t idx) {
  return dim_at_idx(s.dims, idx);
}

} // namespace FlexFlow
