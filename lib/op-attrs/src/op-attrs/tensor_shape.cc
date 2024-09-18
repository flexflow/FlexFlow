#include "op-attrs/tensor_shape.h"
#include "op-attrs/datatype.h"
#include "op-attrs/tensor_dims.h"
#include "utils/containers/get_only.h"
#include "utils/containers/product.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

size_t num_dims(TensorShape const &s) {
  return s.dims.ff_ordered.size();
}

size_t dim_at_idx(TensorShape const &s, ff_dim_t idx) {
  return dim_at_idx(s.dims, idx);
}

size_t &dim_at_idx(TensorShape &s, ff_dim_t idx) {
  return dim_at_idx(s.dims, idx);
}

size_t get_num_elements(TensorShape const &s) {
  return product(s.dims.ff_ordered);
}

size_t get_size_in_bytes(TensorShape const &s) {
  return get_num_elements(s) * size_of_datatype(s.data_type);
}

} // namespace FlexFlow
