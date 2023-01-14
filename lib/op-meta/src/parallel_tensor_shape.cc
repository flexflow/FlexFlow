#include "op-meta/parallel_tensor_shape.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

ParallelTensorShape::ParallelTensorShape(int num_dims,
                                         std::vector<ParallelDim> const &dims,
                                         DataType data_type)
    : num_dims(num_dims), data_type(data_type) {
  for (int i = 0; i < num_dims; i++) {
    this->dims[i] = dims[i];
  }
}

int ParallelTensorShape::get_num_replica_dims() const {
  int num_replica_dims = 0;
  for (int i = 0; i < this->num_dims; i++) {
    if (this->dims[i].is_replica_dim) {
      num_replica_dims++;
    }
  }

  return num_replica_dims;
}

int ParallelTensorShape::get_num_replicas() const {
  int num_replicas = 1;
  for (int i = 0; i < this->num_dims; i++) {
    if (this->dims[i].is_replica_dim) {
      num_replicas *= this->dims[i].degree;
    }
  }

  return num_replicas;
}

std::ostream &operator<<(std::ostream &s, ParallelTensorShape const &shape) {
  s << "[ ";
  for (int i = 0; i < shape.num_dims; i++) {
    s << shape.dims[i].size << "/" << shape.dims[i].degree << " ";
  }
  s << "]";

  return s;
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::ParallelTensorShape>::operator()(
    FlexFlow::ParallelTensorShape const &shape) const {
  size_t key = 0;
  hash_combine(key, shape.num_dims);
  for (int i = 0; i < shape.num_dims; i++) {
    hash_combine(key, 
        shape.dims[i].size, shape.dims[i].degree);
  }
  return key;
}
}; // namespace std
