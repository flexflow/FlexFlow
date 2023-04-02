#include "op-attrs/parallel_tensor_shape.h"
#include "utils/hash-utils.h"
#include "utils/visitable_funcs.h"
#include "utils/containers.h"

VISITABLE_STRUCT(::FlexFlow::ParallelDim, size, degree, parallel_idx, is_replica_dim);
VISITABLE_STRUCT(::FlexFlow::ParallelTensorShape, dims, data_type);

namespace FlexFlow {

ParallelDim::ParallelDim(int size, int degree, int parallel_idx, bool is_replica_dim) 
  : size(size), degree(degree), parallel_idx(parallel_idx), is_replica_dim(is_replica_dim)
{ }

bool ParallelDim::operator==(ParallelDim const &other) const {
  return visit_eq(*this, other);
}

bool ParallelDim::operator!=(ParallelDim const &other) const {
  return visit_neq(*this, other);
}

ParallelTensorShape::ParallelTensorShape(std::vector<ParallelDim> const &dims,
                                         DataType data_type)
    : dims(dims.cbegin(), dims.cend()), data_type(data_type) { }

int ParallelTensorShape::get_num_replica_dims() const {
  int num_replica_dims = 0;
  for (ParallelDim const &dim : this->dims) {
    if (dim.is_replica_dim) {
      num_replica_dims++;
    }
  }

  return num_replica_dims;
}

int ParallelTensorShape::get_num_replicas() const {
  int num_replicas = 1;
  for (ParallelDim const &dim : this->dims) {
    if (dim.is_replica_dim) {
      num_replicas &= dim.degree;
    }
  }

  return num_replicas;
}

std::ostream &operator<<(std::ostream &s, ParallelTensorShape const &shape) {
  s << "[ ";
  for (ParallelDim const &dim : shape) {
    s << dim.size << "/" << dim.degree << " ";
  };
  s << "]";

  return s;
}

ParallelDim const &ParallelTensorShape::at(int index) const {
  return this->dims.at(index);
}

ParallelDim &ParallelTensorShape::at(int index) {
  return this->dims.at(index);
}

size_t ParallelTensorShape::size() const {
  return this->dims.size();
}

size_t ParallelTensorShape::num_dims() const {
  return this->dims.size();
}

ParallelTensorShape::iterator ParallelTensorShape::begin() { return this->dims.begin(); };
ParallelTensorShape::const_iterator ParallelTensorShape::begin() const { return this->cbegin(); };
ParallelTensorShape::const_iterator ParallelTensorShape::cbegin() const { return this->dims.cbegin(); };

ParallelTensorShape::iterator ParallelTensorShape::end() { return this->dims.end(); };
ParallelTensorShape::const_iterator ParallelTensorShape::end() const { return this->cend(); };
ParallelTensorShape::const_iterator ParallelTensorShape::cend() const { return this->dims.cend(); };

}

namespace std {
using ::FlexFlow::ParallelDim;
using ::FlexFlow::ParallelTensorShape;

size_t hash<ParallelDim>::operator()(ParallelDim const &dim) const {
  return visit_hash(dim);
}

size_t hash<ParallelTensorShape>::operator()(
    ParallelTensorShape const &shape) const {
  size_t key = 0;
  hash_combine(key, shape.size());
  for (ParallelDim const &dim : shape) {
    hash_combine(key, dim);
  }
  return key;
}
}
