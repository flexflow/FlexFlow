#include "kernels/softmax_kernels.h"
#include "kernels/softmax_kernels_cpu.h"
#include "kernels/softmax_kernels_gpu.h"
#include "utils/optional.h"

namespace FlexFlow {

std::optional<SoftmaxPerDeviceState>
    softmax_init_kernel(DeviceType device_type,
                        SoftmaxAttrs const &attrs,
                        TensorShape const &input_shape,
                        TensorShape const &output_shape) {
  if (device_type == DeviceType::GPU) {
    return softmax_gpu_init_kernel(
        /*attrs=*/attrs,
        /*input_shape=*/input_shape,
        /*output_shape=*/output_shape);
  } else {
    ASSERT(device_type == DeviceType::CPU);
    return std::nullopt;
  }
}

void softmax_forward_kernel(
    device_stream_t const &stream,
    device_handle_t const &handle,
    std::optional<SoftmaxPerDeviceState> const &per_device_state,
    SoftmaxAttrs const &attrs,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &output) {
  if (stream.is_gpu()) {
    softmax_gpu_forward_kernel(
        /*stream=*/stream.require_gpu(),
        /*handle=*/handle.require_for_gpu(),
        /*per_device_state=*/assert_unwrap(per_device_state),
        /*attrs=*/attrs,
        /*input=*/input,
        /*output=*/output);
  } else {
    ASSERT(stream.is_cpu());
    ASSERT(handle.is_for_cpu());
    ASSERT(!per_device_state.has_value());
    softmax_cpu_forward_kernel(
        /*attrs=*/attrs,
        /*input=*/input,
        /*output=*/output);
  }
}

void softmax_backward_kernel(
    device_stream_t const &stream,
    device_handle_t const &handle,
    std::optional<SoftmaxPerDeviceState> const &per_device_state,
    SoftmaxAttrs const &attrs,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &input_grad) {
  if (stream.is_gpu()) {
    softmax_gpu_backward_kernel(
        /*stream=*/stream.require_gpu(),
        /*handle=*/handle.require_for_gpu(),
        /*per_device_state=*/assert_unwrap(per_device_state),
        /*attrs=*/attrs,
        /*output=*/output,
        /*output_grad=*/output_grad,
        /*input=*/input,
        /*input_grad=*/input_grad);
  } else {
    ASSERT(stream.is_cpu());
    ASSERT(handle.is_for_cpu());
    ASSERT(!per_device_state.has_value());
    softmax_cpu_backward_kernel(
        /*attrs=*/attrs,
        /*output=*/output,
        /*output_grad=*/output_grad,
        /*input=*/input,
        /*input_grad=*/input_grad);
  }
}

void softmax_cleanup_kernel(
    DeviceType device_type,
    std::optional<SoftmaxPerDeviceState> &per_device_state) {
  if (device_type == DeviceType::GPU) {
    softmax_gpu_cleanup_kernel(per_device_state.value());
  } else {
    ASSERT(device_type == DeviceType::CPU);
    ASSERT(!per_device_state.has_value());
  }
}

} // namespace FlexFlow
