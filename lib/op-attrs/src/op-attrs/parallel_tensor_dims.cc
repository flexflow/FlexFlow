#include "op-attrs/parallel_tensor_dims.h"
#include "utils/containers.h"

namespace FlexFlow {

std::vector<ParallelDim> as_vector(ParallelTensorDims const &d) {
  return as_vector(d.unwrapped);
}

int get_num_replica_dims(ParallelTensorDims const &d) {
  return count(d.unwrapped, is_replica_dim);
}

bool is_valid(ParallelTensorDims const &dims) {
  return all_of(dims.unwrapped, [](ParallelDim const &d) { return is_valid(d); });
}

}
