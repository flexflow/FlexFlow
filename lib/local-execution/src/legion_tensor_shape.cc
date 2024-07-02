#include "local-execution/legion_tensor_shape.h"

namespace FlexFlow {

legion_dim_t legion_dim_from_ff_dim(ff_dim_t ff_dim, TensorShape const &shape) {
  return legion_dim_t(shape.dims.ff_ordered.size() - ff_dim.value - 1);
}

} // namespace FlexFlow
