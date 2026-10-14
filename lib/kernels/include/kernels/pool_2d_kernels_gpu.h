#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_POOL_2D_KERNELS_GPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_POOL_2D_KERNELS_GPU_H

#include "kernels/device.h"
#include "kernels/ff_handle.h"
#include "kernels/pool_2d_per_device_state.dtg.h"
#include "op-attrs/activation.dtg.h"
#include "op-attrs/ops/pool_2d.h"

namespace FlexFlow {

Pool2DPerDeviceState
    pool2d_gpu_init_kernel(PerDeviceFFHandle handle,
                           std::optional<Activation> activation,
                           int input_w,
                           int input_h,
                           int input_c,
                           int input_n,
                           int output_w,
                           int output_h,
                           int output_c,
                           int output_n,
                           int pad_h,
                           int pad_w,
                           int kernel_h,
                           int kernel_w,
                           int stride_h,
                           int stride_w,
                           PoolOp pool_type);

void pool2d_gpu_forward_kernel(ffStream_t stream,
                               Pool2DPerDeviceState const &per_device_state,
                               void const *input_ptr,
                               void *output_ptr);

void pool2d_gpu_backward_kernel(ffStream_t stream,
                                Pool2DPerDeviceState const &per_device_state,
                                void const *output_ptr,
                                void const *output_grad_ptr,
                                void const *input_ptr,
                                void *input_grad_ptr);

void pool2d_gpu_cleanup_kernel(Pool2DPerDeviceState &per_device_state);

} // namespace FlexFlow

#endif
