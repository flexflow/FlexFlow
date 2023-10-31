#include "op-attrs/parallel_tensor_shape.h"
#include "utils/containers.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

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

TensorShape get_tensor_shape_unsafe(ParallelTensorShape const &shape) {
  std::vector<size_t> parallel_tensor_dims_size;
  for (int i = 0; i < shape.num_dims(); i++) {
    parallel_tensor_dims_size.push_back(shape.at(ff_dim_t(i)).size);
  }
  TensorDims tensorDims(parallel_tensor_dims_size.begin(),
                        parallel_tensor_dims_size.end());
  return TensorShape(tensorDims, shape.data_type);
}

ParallelDim  ParallelTensorShape::at(ff_dim_t const & index) const {
  return dims.at(index);
}

int ParallelTensorShape::num_dims() const {
  return dims.num_dims();
}



std::vector<TensorShape> get_tensor_shapes_unsafe(
    std::vector<ParallelTensorShape> const &parallelshape_vec) {
  std::vector<TensorShape> tensor_shape_vec;
  for (auto const &shape : parallelshape_vec) {
    tensor_shape_vec.push_back(get_tensor_shape_unsafe(shape));
  }
  return tensor_shape_vec;
}

} // namespace FlexFlow
