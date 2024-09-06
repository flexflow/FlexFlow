#include "op-attrs/tensor_dims.h"
#include "op-attrs/dim_ordered/zip.h"
#include "op-attrs/replica_parallel_dim_set.h"
#include "op-attrs/shard_parallel_dim.dtg.h"
#include "utils/containers/all_of.h"
#include "utils/containers/reversed.h"
#include "utils/containers/transform.h"
#include "utils/containers/vector_of.h"
#include "utils/containers/zip.h"
#include "utils/integer_conversions.h"

namespace FlexFlow {

FFOrdered<size_t> const &ff_ordered(TensorDims const &dims) {
  return dims.ff_ordered;
}

size_t num_dims(TensorDims const &dims) {
  return dims.ff_ordered.size();
}

size_t dim_at_idx(TensorDims const &dims, ff_dim_t idx) {
  return dims.ff_ordered.at(idx);
}

size_t &dim_at_idx(TensorDims &dims, ff_dim_t idx) {
  return dims.ff_ordered.at(idx);
}

bool tensor_dims_is_broadcastable_to(TensorDims const &curr,
                                     TensorDims const &goal) {
  if (num_dims(curr) > num_dims(goal)) {
    return false;
  }

  std::vector<size_t> curr_dims = vector_of(curr.ff_ordered);
  std::vector<size_t> goal_dims = vector_of(goal.ff_ordered);

  for (auto const &[curr_dim, goal_dim] :
       zip(reversed(curr_dims), reversed(goal_dims))) {
    if (curr_dim != 1 && curr_dim != goal_dim) {
      return false;
    }
  }

  return true;
}

std::optional<TensorDims>
    get_broadcast_target_dims(std::unordered_set<TensorDims> const &dims) {
  for (TensorDims target_candidate : dims) {
    if (all_of(dims, [&](TensorDims const &d) {
          return tensor_dims_is_broadcastable_to(d, target_candidate);
        })) {
      return target_candidate;
    }
  }

  return std::nullopt;
}

ParallelTensorDims lift_to_parallel(TensorDims const &dims) {
  std::vector<int> shard_degrees(num_dims(dims),
                                 1); // 1 repeated num_dims(dims) times
  return lift_to_parallel_with_degrees(
      dims, SumDegree{1}, DiscardCopyDegree{1}, shard_degrees);
}

ParallelTensorDims
    lift_to_parallel_with_degrees(TensorDims const &dims,
                                  SumDegree sum_degree,
                                  DiscardCopyDegree discard_copy_degree,
                                  FFOrdered<int> const &shard_degrees) {
  std::vector<ShardParallelDim> lifted =
      transform(zip(vector_of(dims.ff_ordered), vector_of(shard_degrees)),
                [](std::pair<size_t, int> const &p) {
                  size_t size = p.first;
                  int degree = p.second;
                  return ShardParallelDim(size, degree);
                });

  return ParallelTensorDims{FFOrdered<ShardParallelDim>{lifted},
                            ReplicaParallelDimSet{
                                sum_degree,
                                discard_copy_degree,
                            }};
}

} // namespace FlexFlow
