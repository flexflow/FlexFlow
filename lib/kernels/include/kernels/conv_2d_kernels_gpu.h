#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_CONV_2D_KERNELS_GPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_CONV_2D_KERNELS_GPU_H

namespace FlexFlow {

Conv2DPerDeviceState
    conv2d_gpu_init_kernel(PerDeviceFFHandle const &handle,
                           std::optional<Activation> const &activation,
                           int kernel_h,
                           int kernel_w,
                           int groups,
                           int padding_h,
                           int padding_w,
                           int stride_h,
                           int stride_w,
                           GenericTensorAccessorW const &input,
                           GenericTensorAccessorW const &output,
                           float const *filter_ptr,
                           float *filter_grad_ptr);

void conv2d_gpu_forward_kernel(ffStream_t stream,
                               Conv2DPerDeviceState const &m,
                               float const *input_ptr,
                               float *output_ptr,
                               float const *filter_ptr,
                               float const *bias_ptr,
                               std::optional<Activation> activation);

void conv2d_gpu_backward_kernel(ffStream_t stream,
                                Conv2DPerDeviceState const &m,
                                float const *output_ptr,
                                float const *output_grad_ptr,
                                float const *input_ptr,
                                float *input_grad_ptr,
                                float const *filter_ptr,
                                float *filter_grad_ptr,
                                float *bias_grad_ptr,
                                std::optional<Activation> activation);

void conv2d_gpu_cleanup_kernel(Conv2DPerDeviceState &per_device_state);

} // namespace FlexFlow

#endif
