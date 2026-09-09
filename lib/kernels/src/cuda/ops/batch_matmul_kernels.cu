#include "internal/device.h"
#include "kernels/batch_matmul_kernels_gpu.h"
#include "op-attrs/ff_ordered/ff_ordered_slice.h"
#include "op-attrs/tensor_dims.h"
#include "utils/containers/product.h"
#include "utils/containers/require_same.h"
#include "utils/containers/vector_of.h"

namespace FlexFlow {

// The dimensions of a batched matmul `output[b] = input_lhs[b] * input_rhs[b]`,
// where input_lhs is (batch_size, lhs_rows, inner), input_rhs is (batch_size,
// inner, rhs_cols) and output is (batch_size, lhs_rows, rhs_cols).
//
// Inputs may have more than one leading (batch) dimension, matching
// @ref batch_matmul_get_output_shape. Since the tensors are contiguous and
// row-major, all leading dimensions fold into a single cuBLAS batch count
// without any data movement.
struct BatchMatmulDims {
  int batch_size;
  int lhs_rows;
  int inner;
  int rhs_cols;
};

// The dimensions preceding the trailing (rows, cols) pair, which together make
// up the batch of matrices being multiplied.
static FFOrdered<positive_int> get_leading_dims(TensorShape const &shape) {
  return ff_ordered_slice(
      shape.dims.ff_ordered, relative_ff_dim_t{0}, relative_ff_dim_t{-2});
}

static BatchMatmulDims get_batch_matmul_dims(TensorShape const &lhs_shape,
                                             TensorShape const &rhs_shape,
                                             TensorShape const &output_shape) {
  ASSERT(get_num_dims(lhs_shape.dims) >= num_tensor_dims_t{3_n},
         "BatchMatmul expects tensors of at least 3 dimensions",
         lhs_shape);

  require_same(lhs_shape.data_type, rhs_shape.data_type);
  ASSERT(require_same(lhs_shape.data_type, output_shape.data_type) ==
             DataType::FLOAT,
         "BatchMatmul currently only supports data_type = FLOAT. "
         "If you need this feature, please create an issue.",
         lhs_shape.data_type);

  FFOrdered<positive_int> leading_dims =
      require_same(get_leading_dims(lhs_shape),
                   get_leading_dims(rhs_shape),
                   get_leading_dims(output_shape));

  positive_int batch_size = product(vector_of(leading_dims));

  positive_int lhs_rows =
      require_same(dim_at_idx(lhs_shape.dims, relative_ff_dim_t{-2}),
                   dim_at_idx(output_shape.dims, relative_ff_dim_t{-2}));

  positive_int inner =
      require_same(dim_at_idx(lhs_shape.dims, relative_ff_dim_t{-1}),
                   dim_at_idx(rhs_shape.dims, relative_ff_dim_t{-2}));

  positive_int rhs_cols =
      require_same(dim_at_idx(rhs_shape.dims, relative_ff_dim_t{-1}),
                   dim_at_idx(output_shape.dims, relative_ff_dim_t{-1}));

  return BatchMatmulDims{
      /*batch_size=*/batch_size.int_from_positive_int(),
      /*lhs_rows=*/lhs_rows.int_from_positive_int(),
      /*inner=*/inner.int_from_positive_int(),
      /*rhs_cols=*/rhs_cols.int_from_positive_int(),
  };
}

void batch_matmul_gpu_forward_kernel(cudaStream_t stream,
                                     PerDeviceFFHandle const &handle,
                                     GenericTensorAccessorR const &input_lhs,
                                     GenericTensorAccessorR const &input_rhs,
                                     GenericTensorAccessorW const &output) {
  BatchMatmulDims dims =
      get_batch_matmul_dims(input_lhs.shape, input_rhs.shape, output.shape);

  checkCUBLAS(cublasSetStream(handle.blas, stream));

  // Our tensors are row-major but cuBLAS is column-major, so we compute
  // `output^T = input_rhs^T * input_lhs^T` instead. A row-major (r, c) matrix
  // is bit-for-bit a column-major (c, r) matrix, so this needs no transposes
  // and no data movement: it is just a matter of swapping the operands.
  float alpha = 1.0f, beta = 0.0f;
  checkCUBLAS(cublasSgemmStridedBatched(handle.blas,
                                        CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        /*m=*/dims.rhs_cols,
                                        /*n=*/dims.lhs_rows,
                                        /*k=*/dims.inner,
                                        &alpha,
                                        /*A=*/input_rhs.get_float_ptr(),
                                        /*lda=*/dims.rhs_cols,
                                        /*strideA=*/dims.inner * dims.rhs_cols,
                                        /*B=*/input_lhs.get_float_ptr(),
                                        /*ldb=*/dims.inner,
                                        /*strideB=*/dims.lhs_rows * dims.inner,
                                        &beta,
                                        /*C=*/output.get_float_ptr(),
                                        /*ldc=*/dims.rhs_cols,
                                        /*strideC=*/
                                        dims.lhs_rows * dims.rhs_cols,
                                        /*batchCount=*/dims.batch_size));
}

void batch_matmul_gpu_backward_kernel(
    cudaStream_t stream,
    PerDeviceFFHandle const &handle,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &input_lhs,
    GenericTensorAccessorW const &input_lhs_grad,
    GenericTensorAccessorR const &input_rhs,
    GenericTensorAccessorW const &input_rhs_grad) {
  require_same(input_lhs.shape, input_lhs_grad.shape);
  require_same(input_rhs.shape, input_rhs_grad.shape);
  require_same(output.shape, output_grad.shape);

  BatchMatmulDims dims = get_batch_matmul_dims(
      input_lhs.shape, input_rhs.shape, output_grad.shape);

  checkCUBLAS(cublasSetStream(handle.blas, stream));

  // NOTE: beta is 0 so that the gradients are overwritten rather than
  // accumulated into, matching batch_matmul_cpu_backward_kernel.
  float alpha = 1.0f, beta = 0.0f;

  // input_lhs_grad = output_grad * input_rhs^T
  checkCUBLAS(cublasSgemmStridedBatched(handle.blas,
                                        CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        /*m=*/dims.inner,
                                        /*n=*/dims.lhs_rows,
                                        /*k=*/dims.rhs_cols,
                                        &alpha,
                                        /*A=*/input_rhs.get_float_ptr(),
                                        /*lda=*/dims.rhs_cols,
                                        /*strideA=*/dims.inner * dims.rhs_cols,
                                        /*B=*/output_grad.get_float_ptr(),
                                        /*ldb=*/dims.rhs_cols,
                                        /*strideB=*/
                                        dims.lhs_rows * dims.rhs_cols,
                                        &beta,
                                        /*C=*/input_lhs_grad.get_float_ptr(),
                                        /*ldc=*/dims.inner,
                                        /*strideC=*/dims.lhs_rows * dims.inner,
                                        /*batchCount=*/dims.batch_size));

  // input_rhs_grad = input_lhs^T * output_grad
  checkCUBLAS(cublasSgemmStridedBatched(handle.blas,
                                        CUBLAS_OP_N,
                                        CUBLAS_OP_T,
                                        /*m=*/dims.rhs_cols,
                                        /*n=*/dims.inner,
                                        /*k=*/dims.lhs_rows,
                                        &alpha,
                                        /*A=*/output_grad.get_float_ptr(),
                                        /*lda=*/dims.rhs_cols,
                                        /*strideA=*/
                                        dims.lhs_rows * dims.rhs_cols,
                                        /*B=*/input_lhs.get_float_ptr(),
                                        /*ldb=*/dims.inner,
                                        /*strideB=*/dims.lhs_rows * dims.inner,
                                        &beta,
                                        /*C=*/input_rhs_grad.get_float_ptr(),
                                        /*ldc=*/dims.rhs_cols,
                                        /*strideC=*/dims.inner * dims.rhs_cols,
                                        /*batchCount=*/dims.batch_size));
}

} // namespace FlexFlow
