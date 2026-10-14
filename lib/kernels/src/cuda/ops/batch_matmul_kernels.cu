#include "internal/device.h"
#include "kernels/upsample_kernels_gpu.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

void batch_matmul_gpu_forward_kernel(ffStream_t stream,
                                     GenericTensorAccessorR const &input_lhs,
                                     GenericTensorAccessorR const &input_rhs,
                                     GenericTensorAccessorW const &output) {
  NOT_IMPLEMENTED();
}

void batch_matmul_gpu_backward_kernel(
    ffStream_t stream,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &input_lhs,
    GenericTensorAccessorW const &input_lhs_grad,
    GenericTensorAccessorR const &input_rhs,
    GenericTensorAccessorW const &input_rhs_grad) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
