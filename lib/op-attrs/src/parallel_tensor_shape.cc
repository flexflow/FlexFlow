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

size_t ParallelTensorDims::get_volume() const {
  NOT_IMPLEMENTED();
}

size_t ParallelTensorDims::num_dims() const {
  NOT_IMPLEMENTED();
}

ParallelDim const &ParallelTensorDims::at(ff_dim_t const &) const {
  NOT_IMPLEMENTED();
}

ParallelDim &ParallelTensorDims::at(ff_dim_t const &) {
  NOT_IMPLEMENTED();
}

ParallelTensorDims::iterator ParallelTensorDims::begin() {
  NOT_IMPLEMENTED();
}

ParallelTensorDims::const_iterator ParallelTensorDims::begin() const {
  NOT_IMPLEMENTED();
}

ParallelTensorDims::const_iterator ParallelTensorDims::cbegin() const {
  NOT_IMPLEMENTED();
}

ParallelTensorDims::iterator ParallelTensorDims::end() {
  NOT_IMPLEMENTED();
}

ParallelTensorDims::const_iterator ParallelTensorDims::end() const {
  NOT_IMPLEMENTED();
}

ParallelTensorDims::const_iterator ParallelTensorDims::cend() const {
  NOT_IMPLEMENTED();
}

ParallelTensorDims::reverse_iterator ParallelTensorDims::rbegin() {
  NOT_IMPLEMENTED();
}

ParallelTensorDims::const_reverse_iterator ParallelTensorDims::rbegin() const {
  NOT_IMPLEMENTED();
}

ParallelTensorDims::const_reverse_iterator ParallelTensorDims::crbegin() const {
  NOT_IMPLEMENTED();
}

ParallelTensorDims::reverse_iterator ParallelTensorDims::rend() {
  NOT_IMPLEMENTED();
}

ParallelTensorDims::const_reverse_iterator ParallelTensorDims::rend() const {
  NOT_IMPLEMENTED();
}

ParallelTensorDims::const_reverse_iterator ParallelTensorDims::crend() const {
  NOT_IMPLEMENTED();
}

ParallelTensorShape::ParallelTensorShape(TensorShape const &tensor_shape)
    : dims(tensor_shape.dims), data_type(tensor_shape.data_type) {}

int ParallelTensorShape::num_dims() const {
  NOT_IMPLEMENTED();
}

ParallelDim const &ParallelTensorShape::at(ff_dim_t const &) const {
  NOT_IMPLEMENTED();
}

ParallelDim &ParallelTensorShape::at(ff_dim_t const &) {
  NOT_IMPLEMENTED();
}

ParallelDim const &ParallelTensorShape::operator[](ff_dim_t const &) const {
  NOT_IMPLEMENTED();
}

ParallelDim &ParallelTensorShape::operator[](ff_dim_t const &) {
  NOT_IMPLEMENTED();
}

TensorShape get_piece_shape(ParallelTensorShape const &) {
  NOT_IMPLEMENTED();
}

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

TensorShape get_tensor_shape_unsafe(ParallelTensorShape const &) {
  NOT_IMPLEMENTED();
}

std::vector<TensorShape>
    get_tensor_shapes_unsafe(std::vector<ParallelTensorShape> const &) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
