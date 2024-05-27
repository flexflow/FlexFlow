#include "op-attrs/ops/linear.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_shape.h"
#include "utils/integer_conversions.h"

namespace FlexFlow {

TensorShape get_kernel_shape(LinearAttrs const &attrs,
                             TensorShape const &input_shape) {
  size_t in_channels = dim_at_idx(input_shape, ff_dim_t{-1});

  return TensorShape{
      TensorDims{
          FFOrdered<size_t>{in_channels, size_t_from_int(attrs.out_channels)},
      },
      input_shape.data_type,
  };
}

TensorShape get_bias_shape(LinearAttrs const &attrs,
                           TensorShape const &input_shape) {
  return TensorShape{
      TensorDims{
          FFOrdered<size_t>{size_t_from_int(attrs.out_channels)},
      },
      input_shape.data_type,
  };
}

TensorShape get_output_shape(LinearAttrs const &attrs,
                             TensorShape const &input_shape) {
  TensorShape output_shape = input_shape;
  output_shape.dims.ff_ordered.at(ff_dim_t{-1}) =
      size_t_from_int(attrs.out_channels);

  return output_shape;
}

ParallelTensorShape get_kernel_shape(LinearAttrs const &attrs,
                                     ParallelTensorShape const &input_shape) {
  NOT_IMPLEMENTED();
  /* ShardParallelDim input_sample_dim = shard_dim_at_idx(input_shape,
   * ff_dim_t{-2}); */
  /* ShardParallelDim in_channels_dim = shard_dim_at_idx(input_shape,
   * ff_dim_t{-1}); */
}

ParallelTensorShape get_output_shape(LinearAttrs const &attrs,
                                     ParallelTensorShape const &input_shape) {
  ShardParallelDim input_sample_dim =
      shard_dim_at_idx(input_shape, ff_dim_t{-2});
  ShardParallelDim in_channels_dim =
      shard_dim_at_idx(input_shape, ff_dim_t{-1});

  ShardParallelDim output_sample_dim = input_sample_dim;
  ShardParallelDim output_channels_dim = {
      size_t_from_int(attrs.out_channels),
      get_discard_copy_degree(input_shape),
  };

  int output_sum_degree =
      get_sum_degree(input_shape) * in_channels_dim.degree;
  int output_discard_copy_degree = 1;

  ParallelTensorShape result = input_shape;
  shard_dim_at_idx(result, ff_dim_t{-2}) = output_sample_dim;
  shard_dim_at_idx(result, ff_dim_t{-1}) = output_channels_dim;
  result.dims.replica_dims.sum_degree = output_sum_degree;
  result.dims.replica_dims.discard_copy_degree = output_discard_copy_degree;

  assert(total_parallel_degree(result.dims) ==
         total_parallel_degree(input_shape.dims));

  return result;
}

} // namespace FlexFlow
