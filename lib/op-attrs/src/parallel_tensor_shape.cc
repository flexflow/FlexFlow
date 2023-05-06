#include "op-attrs/parallel_tensor_shape.h"
#include "utils/hash-utils.h"
#include "utils/containers.h"

namespace FlexFlow {

ParallelDim::ParallelDim(size_t size, int degree, int parallel_idx, bool is_replica_dim) 
  : size(size), degree(degree), parallel_idx(parallel_idx), is_replica_dim(is_replica_dim)
{ }

static std::vector<ParallelDim> lift_dims(TensorDims const &dims) {
  std::vector<ParallelDim> lifted_dims;
  for (size_t dim_size : dims) {
    lifted_dims.push_back({dim_size, 1, -1, false});
  }
  lifted_dims.push_back({1, 1, -1, true});
  return lifted_dims;
}

ParallelTensorDims::ParallelTensorDims(TensorDims const &dims) 
  : FFOrdered<ParallelDim>(lift_dims(dims))
{ }
  

ParallelTensorShape::ParallelTensorShape(TensorShape const &tensor_shape)
  : dims(tensor_shape.dims), data_type(tensor_shape.data_type)
{ }

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

}
