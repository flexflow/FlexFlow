#include "op-attrs/tensor_dims.h"
#include "op-attrs/replica_parallel_dim_set.h"
#include "op-attrs/shard_parallel_dim.dtg.h"
#include "utils/containers/all_of.h"
#include "utils/containers/as_vector.h"
#include "utils/containers/reversed.h"
#include "utils/containers/transform.h"
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

  std::vector<size_t> curr_dims = as_vector(curr.ff_ordered);
  std::vector<size_t> goal_dims = as_vector(goal.ff_ordered);

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

} // namespace FlexFlow
