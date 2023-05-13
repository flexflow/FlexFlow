#include "op-attrs/parallel_tensor_shape.h"
#include "utils/hash-utils.h"
#include "utils/containers.h"

namespace FlexFlow {

ParallelDim::ParallelDim(size_t size, int degree, bool is_replica_dim) 
  : size(size), degree(degree), is_replica_dim(is_replica_dim)
{ }

static std::vector<ParallelDim> lift_dims(TensorDims const &dims) {
  std::vector<ParallelDim> lifted_dims;
  for (size_t dim_size : dims) {
    lifted_dims.push_back({dim_size, 1, false});
  }
  lifted_dims.push_back({1, 1, true});
  return lifted_dims;
}

ParallelTensorDims::ParallelTensorDims(TensorDims const &dims) 
  : FFOrdered<ParallelDim>(lift_dims(dims))
{ }
  

ParallelTensorShape::ParallelTensorShape(TensorShape const &tensor_shape)
  : dims(tensor_shape.dims), data_type(tensor_shape.data_type)
{ }

int get_num_replica_dims(ParallelTensorShape const &shape) {
  return count(shape.dims, is_replica_dim);
}

int get_num_replicas(ParallelTensorShape const &shape) const {
  return product(transform(filter(is_replica_dim, shape.dims), &ParallelDim::degree));
}

bool is_valid(ParallelTensorShape const &shape) {
  return all_of(shape.dims, is_valid) && shape.data_type != DT_NONE;
}

std::unordered_map<int, int>
    ParallelTensorShape::get_mv_dim_to_tensor_dim_mapping() const {
  std::unordered_map<int, int> result;
  for (int i = 0; i < this->num_dims; i++) {
    int machine_view_dim = this->dims[i].parallel_idx;
    if (machine_view_dim != -1) {
      assert(result.find(machine_view_dim) == result.end());
      result[machine_view_dim] = i;
    }
  }
  return result;
}

std::unordered_map<int, int>
    ParallelTensorShape::get_tensor_dim_to_mv_dim_mapping() const {
  std::unordered_map<int, int> result;
  for (auto const &kv : this->get_mv_dim_to_tensor_dim_mapping()) {
    assert(result.find(kv.second) == result.end());
    result[kv.second] = kv.first;
  }
  return result;
}

}
