#include "op-attrs/tensor_dims.h"

namespace FlexFlow {

FFOrdered<size_t> const &ff_ordered(TensorDims const &dims) {
  return dims.ff_ordered;
}

size_t dim_at_idx(TensorDims const &dims, ff_dim_t idx) {
  return dims.ff_ordered.at(idx);
}

}
