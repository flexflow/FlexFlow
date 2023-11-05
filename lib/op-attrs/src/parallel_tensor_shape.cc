#include "op-attrs/parallel_tensor_shape.h"
#include "utils/containers.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

int ParallelTensorShape::num_dims() const {
  return dims.num_dims();
}

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

ParallelTensorDims::iterator ParallelTensorDims::begin() {
  return data.begin();
}

ParallelTensorDims::const_iterator ParallelTensorDims::begin() const {
  return data.begin();
}

ParallelTensorDims::const_iterator ParallelTensorDims::cbegin() const {
  return data.cbegin();
}

ParallelTensorDims::iterator ParallelTensorDims::end() {
  return data.end();
}

ParallelTensorDims::const_iterator ParallelTensorDims::end() const {
  return data.end();
}

ParallelTensorDims::const_iterator ParallelTensorDims::cend() const {
  return data.cend();
}

ParallelDim const &ParallelTensorDims::at(ff_dim_t const &d) const {
  return data.at(d);
}

ParallelDim &ParallelTensorDims::at(ff_dim_t const &d) {
  return data.at(d);
}

size_t ParallelTensorDims::num_dims() const {
  return data.size();
}

ParallelDim const &ParallelTensorShape::at(ff_dim_t const &d) const {
  return dims.at(d);
}

ParallelDim &ParallelTensorShape::at(ff_dim_t const &d) {
  return dims.at(d);
}
ParallelDim const &ParallelTensorShape::operator[](ff_dim_t const &d) const {
  return dims.at(d);
}
ParallelDim &ParallelTensorShape::operator[](ff_dim_t const &d) {
  return dims.at(d);  
}

TensorShape get_tensor_shape_unsafe(ParallelTensorShape const &) {
  NOT_IMPLEMENTED();
}
} // namespace FlexFlow
