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

bool tensor_shape_is_broadcastable_to(TensorShape const &curr, TensorShape const &goal) {
  return tensor_dims_is_broadcastable_to(curr.dims, goal.dims) && curr.data_type == goal.data_type;
}

std::optional<TensorShape> get_broadcast_target_shape(std::unordered_set<TensorShape> const &shapes) {
  std::unordered_set<DataType> datatypes = transform(shapes, [](TensorShape const &s) { return s.data_type; });

  if (datatypes.size() != 1) {
    return std::nullopt;
  }

  std::unordered_set<TensorDims> shapes_dims = transform(shapes, [](TensorShape const &s) { return s.dims; });

  std::optional<TensorDims> maybe_result_dims = get_broadcast_target_dims(shapes_dims);
  std::optional<TensorShape> result = transform(maybe_result_dims, [&](TensorDims const &result_dims) { 
    return TensorShape{
      result_dims,
      get_only(datatypes),
    };
  });
  
  return result;
}

} // namespace FlexFlow
