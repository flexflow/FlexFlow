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

std::vector<size_t> ParallelTensorDims::get_dims() const {
  std::vector<size_t> dims;
  for (ParallelDim const &d : this->data) {
    dims.push_back(d.size);
  }
  return dims;
}

size_t ParallelTensorDims::get_volume() const {

  // this function can use contains.h to optimize the code
  size_t volume = 1;
  for (ParallelDim const &d : this->data) {
    volume *= d.size;
  }
  return volume;
}

ParallelTensorShape::ParallelTensorShape(TensorShape const &tensor_shape)
    : dims(tensor_shape.dims), data_type(tensor_shape.data_type) {}

int get_num_replica_dims(ParallelTensorShape const &shape) {
  return count(shape.dims, is_replica_dim);
}

TensorShape get_piece_shape(ParallelTensorShape const &parall_tensor_shape) {
  return TensorShape(parall_tensor_shape.dims, parall_tensor_shape.data_type);
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
