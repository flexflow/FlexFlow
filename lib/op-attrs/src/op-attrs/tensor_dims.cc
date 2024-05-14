#include "op-attrs/tensor_dims.h"
#include "utils/containers.h"
#include "op-attrs/replica_parallel_dim_set.h"
#include "op-attrs/shard_parallel_dim.dtg.h"

namespace FlexFlow {

FFOrdered<size_t> const &ff_ordered(TensorDims const &dims) {
  return dims.ff_ordered;
}

size_t num_dims(TensorDims const &dims) {
  return dims.ff_ordered.size();
}

size_t dim_at_idx(TensorDims const &dims, ff_dim_t idx) {
  if (idx.value < 0) {
    idx = ff_dim_t{num_dims(dims) + idx.value};
  }
  return dims.ff_ordered.at(idx);
}

ParallelTensorDims lift_to_parallel(TensorDims const &dims) {
  std::vector<ShardParallelDim> lifted = transform(as_vector(dims.ff_ordered), [](size_t size) { return ShardParallelDim{size, 1}; });

  return ParallelTensorDims{
    FFOrdered<ShardParallelDim>{lifted},
    empty_replica_parallel_dim_set(),
  };
}

}
