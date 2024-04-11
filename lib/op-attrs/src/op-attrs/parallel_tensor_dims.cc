#include "op-attrs/parallel_tensor_dims.h"
#include "utils/containers.h"

namespace FlexFlow {

FFOrdered<ParallelDim> const &ff_ordered(ParallelTensorDims const &d) {
  return d.ff_ordered;
}

std::vector<ParallelDim> as_vector(ParallelTensorDims const &d) {
  return as_vector(d.ff_ordered);
}

int get_num_replica_dims(ParallelTensorDims const &d) {
  return count(d.ff_ordered, is_replica_dim);
}

bool is_valid(ParallelTensorDims const &dims) {
  return all_of(dims.ff_ordered, [](ParallelDim const &d) { return is_valid(d); });
}

size_t num_dims(ParallelTensorDims const &dims) {
  return dims.ff_ordered.size();
}

ParallelDim dim_at_idx(ParallelTensorDims const &d, ff_dim_t idx) {
  return d.ff_ordered.at(idx);
}

ParallelDim &dim_at_idx(ParallelTensorDims &d, ff_dim_t idx) {
  return d.ff_ordered.at(idx);
}

}
