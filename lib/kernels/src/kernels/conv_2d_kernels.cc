#include "kernels/conv_2d_kernels.h"
#include "kernels/conv_2d_kernels_cpu.h"
#include "kernels/conv_2d_kernels_gpu.h"

namespace FlexFlow {

std::optional<Conv2DPerDeviceState>
    conv2d_init_kernel(DeviceType device_type,
                       device_handle_t const &handle,
                       std::optional<Activation> activation,
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
                       float *filter_grad_ptr) {
  if (device_type == DeviceType::GPU) {
    return conv2d_gpu_init_kernel(
        /*handle=*/handle.require_for_gpu(),
        /*activation=*/activation,
        /*kernel_h=*/kernel_h,
        /*kernel_w=*/kernel_w,
        /*groups=*/groups,
        /*padding_h=*/padding_h,
        /*padding_w=*/padding_w,
        /*stride_h=*/stride_h,
        /*stride_w=*/stride_w,
        /*input=*/input,
        /*output=*/output,
        /*filter_ptr=*/filter_ptr,
        /*filter_grad_ptr=*/filter_grad_ptr);
  } else {
    ASSERT(device_type == DeviceType::CPU);
    ASSERT(handle.is_for_cpu());
    return std::nullopt;
  }
}

void conv2d_forward_kernel(
    device_stream_t const &stream,
    std::optional<Conv2DPerDeviceState> const &per_device_state,
    Conv2DAttrs const &attrs,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &output,
    GenericTensorAccessorR const &filter,
    std::optional<GenericTensorAccessorR> const &bias) {
  if (stream.is_gpu()) {
    conv2d_gpu_forward_kernel(
        /*stream=*/stream.require_gpu(),
        /*per_device_state=*/per_device_state.value(),
        /*input_ptr=*/input.get_float_ptr(),
        /*output_ptr=*/output.get_float_ptr(),
        /*filter_ptr=*/filter.get_float_ptr(),
        /*bias=*/
        transform(bias,
                  [&](GenericTensorAccessorR const &t) -> float const * {
                    return t.get_float_ptr();
                  })
            .value_or(nullptr),
        /*activation=*/attrs.activation);
  } else {
    ASSERT(stream.is_cpu());
    conv2d_cpu_forward_kernel(
        /*attrs=*/attrs,
        /*input=*/input,
        /*filter=*/filter,
        /*bias=*/bias,
        /*output=*/output);
  }
}

void conv2d_backward_kernel(
    device_stream_t const &stream,
    std::optional<Conv2DPerDeviceState> const &per_device_state,
    Conv2DAttrs const &attrs,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &input_grad,
    GenericTensorAccessorR const &filter,
    GenericTensorAccessorW const &filter_grad,
    std::optional<GenericTensorAccessorR> const &bias,
    std::optional<GenericTensorAccessorW> const &bias_grad) {
  if (stream.is_gpu()) {
    conv2d_gpu_backward_kernel(
        /*stream=*/stream.require_gpu(),
        /*per_device_state=*/per_device_state.value(),
        /*output_ptr=*/output.get_float_ptr(),
        /*output_grad_ptr=*/output_grad.get_float_ptr(),
        /*input_ptr=*/input.get_float_ptr(),
        /*input_grad_ptr=*/input_grad.get_float_ptr(),
        /*filter_ptr=*/filter.get_float_ptr(),
        /*filter_grad_ptr=*/filter_grad.get_float_ptr(),
        /*bias_grad_ptr=*/
        transform(bias_grad,
                  [&](GenericTensorAccessorW const &t) -> float * {
                    return t.get_float_ptr();
                  })
            .value_or(nullptr),
        /*activation=*/attrs.activation);
  } else {
    ASSERT(stream.is_cpu());
    conv2d_cpu_backward_kernel(
        /*attrs=*/attrs,
        /*input=*/input,
        /*input_grad=*/input_grad,
        /*filter=*/filter,
        /*filter_grad=*/filter_grad,
        /*bias=*/bias,
        /*bias_grad=*/bias_grad,
        /*output=*/output,
        /*output_grad=*/output_grad);
  }
}

void conv2d_cleanup_kernel(
    DeviceType device_type,
    std::optional<Conv2DPerDeviceState> &per_device_state) {
  if (device_type == DeviceType::GPU) {
    conv2d_gpu_cleanup_kernel(per_device_state.value());
  } else {
    ASSERT(device_type == DeviceType::CPU);
    ASSERT(per_device_state == std::nullopt);
  }
}

} // namespace FlexFlow
