#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_BATCH_MATMUL_KERNELS_GPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_BATCH_MATMUL_KERNELS_GPU_H

#include "kernels/accessor.h"
#include "kernels/device.h"
#include "op-attrs/ops/batch_matmul_attrs.dtg.h"

namespace FlexFlow {

void batch_matmul_gpu_forward_kernel(ffStream_t stream,
                                     PerDeviceFFHandle const &handle,
                                     GenericTensorAccessorR const &input_lhs,
                                     GenericTensorAccessorR const &input_rhs,
                                     GenericTensorAccessorW const &output);

void batch_matmul_gpu_backward_kernel(
    ffStream_t stream,
    PerDeviceFFHandle const &handle,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &input_lhs,
    GenericTensorAccessorW const &input_lhs_grad,
    GenericTensorAccessorR const &input_rhs,
    GenericTensorAccessorW const &input_rhs_grad);

} // namespace FlexFlow

#endif
