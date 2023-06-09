#include "op-attrs/parallel_tensor_shape.h"
#include "utils/containers.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

static std::vector<ParallelDim> lift_dims(TensorDims const &dims) {
  std::vector<ParallelDim> lifted_dims;
  for (size_t dim_size : dims) {
    lifted_dims.push_back({dim_size, 1, false});
  }
  lifted_dims.push_back({1, 1, true});
  return lifted_dims;
}

ParallelTensorDims::ParallelTensorDims(TensorDims const &dims)
    : data(lift_dims(dims)) {}

ParallelTensorShape::ParallelTensorShape(TensorShape const &tensor_shape)
    : dims(tensor_shape.dims), data_type(tensor_shape.data_type) {}

int get_num_replica_dims(ParallelTensorShape const &shape) {
  return count(shape.dims, is_replica_dim);
}

int get_num_replicas(ParallelTensorShape const &shape) {
  return product(
      transform(filter(as_vector(shape.dims), is_replica_dim),
                [](ParallelDim const &d) -> int { return d.degree; }));
}

bool is_valid(ParallelTensorDims const &dims) {
  return all_of(dims, [](ParallelDim const &d) { return is_valid(d); });
}

bool is_valid(ParallelTensorShape const &shape) {
  return is_valid(shape.dims);
}

} // namespace FlexFlow
