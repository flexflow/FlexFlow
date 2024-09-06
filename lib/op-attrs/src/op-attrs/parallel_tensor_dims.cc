#include "op-attrs/parallel_tensor_dims.h"
#include "op-attrs/dim_ordered/transform.h"
#include "op-attrs/replica_parallel_dim.h"
#include "op-attrs/replica_parallel_dim_set.h"
#include "op-attrs/shard_parallel_dim.h"
#include "op-attrs/tensor_dims.h"
#include "utils/containers/all_of.h"
#include "utils/containers/as_vector.h"
#include "utils/containers/product.h"
#include "utils/containers/transform.h"
#include "utils/integer_conversions.h"
#include "op-attrs/dim_ordered/zip.h"

namespace FlexFlow {

FFOrdered<ShardParallelDim> ff_ordered_shard_dims(ParallelTensorDims const &d) {
  return d.shard_dims;
}

FFOrdered<int> ff_ordered_shard_degrees(ParallelTensorDims const &d) {
  return transform(d.shard_dims,
                   [](ShardParallelDim const &d) { return d.degree; });
}

std::unordered_set<ReplicaParallelDim>
    replica_dims(ParallelTensorDims const &d) {
  return get_replica_dims(d.replica_dims);
}

size_t num_shard_dims(ParallelTensorDims const &dims) {
  return dims.shard_dims.size();
}

ParallelTensorDimDegrees get_parallel_degrees(ParallelTensorDims const &d) {
  return ParallelTensorDimDegrees{
    d.replica_dims.sum_degree,
    d.replica_dims.discard_copy_degree,
    ff_ordered_shard_degrees(d),
  };
}

ParallelTensorDims lift_to_parallel(TensorDims const &dims) {
  std::vector<int> shard_degrees(num_dims(dims),
                                 1); // 1 repeated num_dims(dims) times
  return lift_to_parallel_with_degrees(
      dims, SumDegree{1}, DiscardCopyDegree{1}, shard_degrees);
}

ParallelTensorDims lift_to_parallel_with_degrees(TensorDims const &unpar,
                                                 SumDegree const &sum_degree,
                                                 DiscardCopyDegree const &discard_copy_degree,
                                                 FFOrdered<int> const &shard_degrees) {
  std::vector<ShardParallelDim> lifted =
      transform(zip(as_vector(unpar.ff_ordered), as_vector(shard_degrees)),
                [](std::pair<size_t, int> const &p) {
                  size_t size = p.first;
                  int degree = p.second;
                  return ShardParallelDim{size, degree};
                });

  return ParallelTensorDims{FFOrdered<ShardParallelDim>{lifted},
                            ReplicaParallelDimSet{
                                sum_degree,
                                discard_copy_degree,
                            }};
}

ParallelTensorDims lift_to_parallel_with_degrees(TensorDims const &unpar,
                                                 ParallelTensorDimDegrees const &degrees) {
  return lift_to_parallel_with_degrees(unpar, degrees.sum_degree, degrees.discard_copy_degree, degrees.shard_degrees);
}


int total_replica_degree(ParallelTensorDims const &dims) {
  return dims.replica_dims.discard_copy_degree.value *
         dims.replica_dims.sum_degree.value;
}

int total_shard_degree(ParallelTensorDims const &dims) {
  return product(transform(as_vector(dims.shard_dims),
                           [](ShardParallelDim const &d) { return d.degree; }));
}

int total_parallel_degree(ParallelTensorDims const &dims) {
  return total_replica_degree(dims) * total_shard_degree(dims);
}

bool is_valid(ParallelTensorDims const &dims) {
  return all_of(dims.shard_dims,
                [](ShardParallelDim const &d) { return is_valid(d); }) &&
         all_of(replica_dims(dims),
                [](ReplicaParallelDim const &d) { return is_valid(d); });
}

ShardParallelDim shard_dim_at_idx(ParallelTensorDims const &d, ff_dim_t idx) {
  return d.shard_dims.at(idx);
}

ShardParallelDim &shard_dim_at_idx(ParallelTensorDims &d, ff_dim_t idx) {
  return d.shard_dims.at(idx);
}

TensorDims get_piece_dims(ParallelTensorDims const &) {
  NOT_IMPLEMENTED();
}

TensorDims get_tensor_dims_unsafe(ParallelTensorDims const &) {
  NOT_IMPLEMENTED();
}

TensorDims get_reduced_dims(ParallelTensorDims const &dims) {
  FFOrdered<size_t> dim_sizes = transform(
      dims.shard_dims, [](ShardParallelDim const &d) { return d.size; });
  return TensorDims{dim_sizes};
}

} // namespace FlexFlow
