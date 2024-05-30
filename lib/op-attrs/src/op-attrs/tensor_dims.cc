#include "op-attrs/tensor_dims.h"
#include "op-attrs/replica_parallel_dim_set.h"
#include "op-attrs/shard_parallel_dim.dtg.h"
#include "utils/containers.h"
#include "utils/containers/zip_vectors.h"
#include "utils/integer_conversions.h"

namespace FlexFlow {

FFOrdered<size_t> const &ff_ordered(TensorDims const &dims) {
  return dims.ff_ordered;
}

size_t num_dims(TensorDims const &dims) {
  return dims.ff_ordered.size();
}

size_t dim_at_idx(TensorDims const &dims, ff_dim_t idx) {
  if (idx.value < 0) {
    idx = ff_dim_t{int_from_size_t(num_dims(dims)) + idx.value};
  }
  return dims.ff_ordered.at(idx);
}

size_t &dim_at_idx(TensorDims &dims, ff_dim_t idx) {
  if (idx.value < 0) {
    idx = ff_dim_t{int_from_size_t(num_dims(dims)) + idx.value};
  }
  return dims.ff_ordered.at(idx);
}

ParallelTensorDims lift_to_parallel(TensorDims const &dims) {
  std::vector<int> shard_degrees(num_dims(dims), 1); // 1 repeated num_dims(dims) times
  return lift_to_parallel_with_degrees(dims, 1, 1, shard_degrees);
}

ParallelTensorDims lift_to_parallel_with_degrees(TensorDims const &dims, SumDegree sum_degree, DiscardCopyDegree discard_copy_degree, FFOrdered<int> const &shard_degrees) {
  std::vector<ShardParallelDim> lifted =
      transform(zip(as_vector(dims.ff_ordered), as_vector(shard_degrees)), 
                [](std::pair<size_t, int> const &p) {
                  size_t size = p.first;
                  int degree = p.second;
                  return ShardParallelDim(size, degree);
                });

  return ParallelTensorDims{
    FFOrdered<ShardParallelDim>{lifted},
    ReplicaParallelDimSet{
      sum_degree,
      discard_copy_degree,
    }
  };
}

} // namespace FlexFlow
