#include "kernels/conv_2d_kernels.h"
#include "kernels/conv_2d_kernels_cpu.h"
#include "kernels/conv_2d_kernels_gpu.h"
#include "utils/optional.h"

namespace FlexFlow {

std::optional<Conv2DPerDeviceState>
    conv_2d_init_kernel(DeviceType device_type,
                        device_handle_t const &handle,
                        Conv2DAttrs const &attrs,
                        TensorShape const &input_shape,
                        TensorShape const &output_shape) {
  if (device_type == DeviceType::GPU) {
    return conv_2d_gpu_init_kernel(
        /*handle=*/handle.require_for_gpu(),
        /*attrs=*/attrs,
        /*input_shape=*/input_shape,
        /*output_shape=*/output_shape);
  } else {
    ASSERT(device_type == DeviceType::CPU);
    ASSERT(handle.is_for_cpu());
    return std::nullopt;
  }
}

void conv_2d_forward_kernel(
    device_stream_t const &stream,
    device_handle_t const &handle,
    std::optional<Conv2DPerDeviceState> const &per_device_state,
    Conv2DAttrs const &attrs,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorR const &filter,
    std::optional<GenericTensorAccessorR> const &bias,
    GenericTensorAccessorW const &output) {
  if (stream.is_gpu()) {
    conv_2d_gpu_forward_kernel(
        /*stream=*/stream.require_gpu(),
        /*handle=*/handle.require_for_gpu(),
        /*per_device_state=*/assert_unwrap(per_device_state),
        /*attrs=*/attrs,
        /*input=*/input,
        /*filter=*/filter,
        /*bias=*/bias,
        /*output=*/output);
  } else {
    ASSERT(stream.is_cpu());
    ASSERT(handle.is_for_cpu());
    ASSERT(!per_device_state.has_value());
    conv_2d_cpu_forward_kernel(
        /*attrs=*/attrs,
        /*input=*/input,
        /*filter=*/filter,
        /*bias=*/bias,
        /*output=*/output);
  }
}

void conv_2d_backward_kernel(
    device_stream_t const &stream,
    device_handle_t const &handle,
    std::optional<Conv2DPerDeviceState> const &per_device_state,
    Conv2DAttrs const &attrs,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &input_grad,
    GenericTensorAccessorR const &filter,
    GenericTensorAccessorW const &filter_grad,
    std::optional<GenericTensorAccessorW> const &bias_grad) {
  if (stream.is_gpu()) {
    conv_2d_gpu_backward_kernel(
        /*stream=*/stream.require_gpu(),
        /*handle=*/handle.require_for_gpu(),
        /*per_device_state=*/assert_unwrap(per_device_state),
        /*attrs=*/attrs,
        /*output=*/output,
        /*output_grad=*/output_grad,
        /*input=*/input,
        /*input_grad=*/input_grad,
        /*filter=*/filter,
        /*filter_grad=*/filter_grad,
        /*bias_grad=*/bias_grad);
  } else {
    ASSERT(stream.is_cpu());
    ASSERT(handle.is_for_cpu());
    ASSERT(!per_device_state.has_value());
    conv_2d_cpu_backward_kernel(
        /*attrs=*/attrs,
        /*output=*/output,
        /*output_grad=*/output_grad,
        /*input=*/input,
        /*input_grad=*/input_grad,
        /*filter=*/filter,
        /*filter_grad=*/filter_grad,
        /*bias_grad=*/bias_grad);
  }
}

void conv_2d_cleanup_kernel(
    DeviceType device_type,
    std::optional<Conv2DPerDeviceState> &per_device_state) {
  if (device_type == DeviceType::GPU) {
    conv_2d_gpu_cleanup_kernel(per_device_state.value());
  } else {
    ASSERT(device_type == DeviceType::CPU);
    ASSERT(!per_device_state.has_value());
  }
}

} // namespace FlexFlow
