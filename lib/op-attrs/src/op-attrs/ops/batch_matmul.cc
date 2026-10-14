#include "op-attrs/ops/batch_matmul.h"
#include "op-attrs/ff_ordered/ff_ordered_concat.h"
#include "op-attrs/ff_ordered/ff_ordered_slice.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "op-attrs/parallel_tensor_dims.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_dims.h"
#include "utils/containers/require_same.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

TensorShape batch_matmul_get_output_shape(BatchMatmulAttrs const &,
                                          TensorShape const &lhs,
                                          TensorShape const &rhs) {
  positive_int lhs_row_dim = dim_at_idx(lhs.dims, relative_ff_dim_t{-2});
  positive_int lhs_col_dim = dim_at_idx(lhs.dims, relative_ff_dim_t{-1});

  positive_int rhs_row_dim = dim_at_idx(rhs.dims, relative_ff_dim_t{-2});
  positive_int rhs_col_dim = dim_at_idx(rhs.dims, relative_ff_dim_t{-1});

  ASSERT(lhs_col_dim == rhs_row_dim);

  auto get_leading_dims = [](TensorShape const &s) -> FFOrdered<positive_int> {
    return ff_ordered_slice(
        s.dims.ff_ordered, relative_ff_dim_t{0}, relative_ff_dim_t{-2});
  };

  FFOrdered<positive_int> leading_dims =
      require_same(get_leading_dims(lhs), get_leading_dims(rhs));

  return TensorShape{
      TensorDims{
          ff_ordered_concat(leading_dims,
                            FFOrdered<positive_int>{
                                lhs_row_dim,
                                rhs_col_dim,
                            }),
      },
      /*data_type=*/require_same(lhs.data_type, rhs.data_type),
  };
}

ParallelTensorDimDegrees batch_matmul_get_output_parallel_dim_degrees(
    BatchMatmulAttrs const &attrs,
    ParallelTensorDimDegrees const &lhs,
    ParallelTensorDimDegrees const &rhs) {
  ASSERT(get_ptensor_dim_degrees_num_shard_dims(lhs) ==
         num_ptensor_shard_dims_t{3_n});
  ASSERT(get_ptensor_dim_degrees_num_shard_dims(rhs) ==
         num_ptensor_shard_dims_t{3_n});

  positive_int batch_degree = require_same(
      get_degree_for_parallel_tensor_dim_idx(lhs, shard_dim_idx(ff_dim_t{0_n})),
      get_degree_for_parallel_tensor_dim_idx(rhs,
                                             shard_dim_idx(ff_dim_t{0_n})));

  positive_int reduction_parallelism_degree = require_same(
      get_degree_for_parallel_tensor_dim_idx(lhs, shard_dim_idx(ff_dim_t{2_n})),
      get_degree_for_parallel_tensor_dim_idx(rhs,
                                             shard_dim_idx(ff_dim_t{1_n})));

  positive_int lhs_row_degree =
      get_degree_for_parallel_tensor_dim_idx(lhs, shard_dim_idx(ff_dim_t{1_n}));

  positive_int rhs_column_degree =
      get_degree_for_parallel_tensor_dim_idx(rhs, shard_dim_idx(ff_dim_t{2_n}));

  ASSERT(lhs_row_degree * lhs.sum_degree.value ==
         rhs.discard_copy_degree.value);

  ASSERT(rhs_column_degree * rhs.sum_degree.value ==
         lhs.discard_copy_degree.value);

  return ParallelTensorDimDegrees{
      /*sum_degree=*/SumDegree{
          reduction_parallelism_degree * lhs.sum_degree.value *
              rhs.sum_degree.value,
      },
      /*discard_copy_degree=*/DiscardCopyDegree{1_p},
      /*shard_degrees=*/
      FFOrdered<positive_int>{
          batch_degree,
          lhs_row_degree,
          rhs_column_degree,
      },
  };
}

ParallelTensorShape
    batch_matmul_get_output_parallel_shape(BatchMatmulAttrs const &attrs,
                                           ParallelTensorShape const &lhs,
                                           ParallelTensorShape const &rhs) {
  TensorShape output_shape = batch_matmul_get_output_shape(
      attrs, get_reduced_shape(lhs), get_reduced_shape(rhs));

  ParallelTensorDimDegrees output_degrees =
      batch_matmul_get_output_parallel_dim_degrees(
          attrs, get_parallel_degrees(lhs), get_parallel_degrees(rhs));

  return lift_to_parallel_with_degrees(output_shape, output_degrees);
}

} // namespace FlexFlow
