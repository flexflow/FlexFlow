
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "utils/containers/extend.h"
#include "utils/containers/range.h"
#include "utils/containers/transform.h"
#include "utils/overload.h"

namespace FlexFlow {

ParallelDim get_parallel_dim_at_idx(ParallelTensorShape const &shape,
                                    parallel_tensor_dim_idx_t idx) {
  return idx.visit<ParallelDim>(
      overload{[&](ff_dim_t shard_dim) {
                 return ParallelDim{shape.dims.shard_dims.at(shard_dim)};
               },
               [&](ReplicaType replica_type) {
                 ReplicaParallelDimSet replicas = shape.dims.replica_dims;
                 int degree = (ReplicaType::SUM == replica_type
                                   ? replicas.sum_degree.value
                                   : replicas.discard_copy_degree.value);
                 return ParallelDim{ReplicaParallelDim{degree, replica_type}};
               }});
}

std::unordered_set<parallel_tensor_dim_idx_t>
    get_parallel_tensor_indices(ParallelTensorShape const &shape) {
  std::unordered_set<parallel_tensor_dim_idx_t> indices;
  extend(indices, transform(range(num_shard_dims(shape)), [](int idx) {
           return parallel_tensor_dim_idx_t(ff_dim_t(idx));
         }));
  indices.insert(parallel_tensor_dim_idx_t(ReplicaType::SUM));
  indices.insert(parallel_tensor_dim_idx_t(ReplicaType::DISCARD_COPY));
  return indices;
}

} // namespace FlexFlow
