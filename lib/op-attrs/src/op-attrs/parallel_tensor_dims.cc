#include "op-attrs/parallel_tensor_dims.h"
#include "op-attrs/ff_ordered/ff_ordered_transform.h"
#include "op-attrs/ff_ordered/ff_ordered_zip.h"
#include "op-attrs/replica_parallel_dim.h"
#include "op-attrs/replica_parallel_dim_set.h"
#include "op-attrs/shard_parallel_dim.h"
#include "op-attrs/tensor_dims.h"
#include "utils/containers/all_of.h"
#include "utils/containers/product.h"
#include "utils/containers/repeat_element.h"
#include "utils/containers/transform.h"
#include "utils/containers/vector_of.h"
#include "utils/integer_conversions.h"
#include "utils/nonnegative_int/num_elements.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

FFOrdered<ShardParallelDim> ff_ordered_shard_dims(ParallelTensorDims const &d) {
  return d.shard_dims;
}

FFOrdered<positive_int> ff_ordered_shard_degrees(ParallelTensorDims const &d) {
  return ff_ordered_transform(
      d.shard_dims, [](ShardParallelDim const &d) { return d.degree; });
}

std::set<ReplicaParallelDim> replica_dims(ParallelTensorDims const &d) {
  return get_replica_dims(d.replica_dims);
}

num_ptensor_shard_dims_t num_shard_dims(ParallelTensorDims const &dims) {
  return num_ptensor_shard_dims_t{
      num_elements(dims.shard_dims),
  };
}

ParallelTensorDimDegrees get_parallel_degrees(ParallelTensorDims const &d) {
  return ParallelTensorDimDegrees{
      d.replica_dims.sum_degree,
      d.replica_dims.discard_copy_degree,
      ff_ordered_shard_degrees(d),
  };
}

ParallelTensorDims lift_to_parallel(TensorDims const &dims) {
  std::vector<positive_int> shard_degrees = repeat_element(
      /*num_times=*/get_num_dims(dims).nonnegative_int_from_num_tensor_dims(),
      /*element=*/1_p);
  return lift_to_parallel_with_degrees(dims,
                                       SumDegree{1_p},
                                       DiscardCopyDegree{1_p},
                                       ff_ordered_of(shard_degrees));
}

ParallelTensorDims lift_to_parallel_with_degrees(
    TensorDims const &unpar,
    SumDegree const &sum_degree,
    DiscardCopyDegree const &discard_copy_degree,
    FFOrdered<positive_int> const &shard_degrees) {
  std::vector<ShardParallelDim> lifted =
      transform(zip(vector_of(unpar.ff_ordered), vector_of(shard_degrees)),
                [](std::pair<positive_int, positive_int> const &p) {
                  positive_int size = p.first;
                  positive_int degree = p.second;
                  return ShardParallelDim{size, degree};
                });

  return ParallelTensorDims{ff_ordered_of(lifted),
                            ReplicaParallelDimSet{
                                sum_degree,
                                discard_copy_degree,
                            }};
}

ParallelTensorDims
    lift_to_parallel_with_degrees(TensorDims const &unpar,
                                  ParallelTensorDimDegrees const &degrees) {
  return lift_to_parallel_with_degrees(unpar,
                                       degrees.sum_degree,
                                       degrees.discard_copy_degree,
                                       degrees.shard_degrees);
}

positive_int total_replica_degree(ParallelTensorDims const &dims) {
  return dims.replica_dims.discard_copy_degree.value *
         dims.replica_dims.sum_degree.value;
}

positive_int total_shard_degree(ParallelTensorDims const &dims) {
  return product(transform(vector_of(dims.shard_dims),
                           [](ShardParallelDim const &d) { return d.degree; }));
}

positive_int total_parallel_degree(ParallelTensorDims const &dims) {
  return total_replica_degree(dims) * total_shard_degree(dims);
}

bool is_valid(ParallelTensorDims const &dims) {
  return all_of(dims.shard_dims,
                [](ShardParallelDim const &d) { return is_valid(d); }) &&
         all_of(replica_dims(dims),
                [](ReplicaParallelDim const &d) { return is_valid(d); });
}

ShardParallelDim shard_dim_at_idx(ParallelTensorDims const &d,
                                  relative_ff_dim_t idx) {
  return d.shard_dims.at(idx);
}

ShardParallelDim &shard_dim_at_idx(ParallelTensorDims &d,
                                   relative_ff_dim_t idx) {
  return d.shard_dims.at(idx);
}

TensorDims get_piece_dims(ParallelTensorDims const &) {
  NOT_IMPLEMENTED();
}

TensorDims get_tensor_dims_unsafe(ParallelTensorDims const &) {
  NOT_IMPLEMENTED();
}

TensorDims get_reduced_dims(ParallelTensorDims const &dims) {
  FFOrdered<positive_int> dim_sizes = ff_ordered_transform(
      dims.shard_dims, [](ShardParallelDim const &d) { return d.size; });
  return TensorDims{dim_sizes};
}

} // namespace FlexFlow
