#include "kernels/batch_norm_kernels.h"
#include "kernels/batch_norm_kernels_cpu.h"
#include "kernels/batch_norm_kernels_gpu.h"
#include "utils/optional.h"

namespace FlexFlow {

std::optional<BatchNormPerDeviceState>
    batch_norm_init_kernel(DeviceType device_type,
                           Allocator &allocator,
                           BatchNormAttrs const &attrs,
                           TensorShape const &input_shape,
                           TensorShape const &output_shape) {
  if (device_type == DeviceType::GPU) {
    return batch_norm_gpu_init_kernel(
        /*allocator=*/allocator,
        /*attrs=*/attrs,
        /*input_shape=*/input_shape,
        /*output_shape=*/output_shape);
  } else {
    ASSERT(device_type == DeviceType::CPU);
    return std::nullopt;
  }
}

void batch_norm_forward_kernel(
    device_stream_t const &stream,
    device_handle_t const &handle,
    std::optional<BatchNormPerDeviceState> const &per_device_state,
    BatchNormAttrs const &attrs,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorR const &gamma,
    GenericTensorAccessorR const &beta,
    GenericTensorAccessorW const &output) {
  if (stream.is_gpu()) {
    batch_norm_gpu_forward_kernel(
        /*stream=*/stream.require_gpu(),
        /*handle=*/handle.require_for_gpu(),
        /*per_device_state=*/assert_unwrap(per_device_state),
        /*attrs=*/attrs,
        /*input=*/input,
        /*gamma=*/gamma,
        /*beta=*/beta,
        /*output=*/output);
  } else {
    ASSERT(stream.is_cpu());
    ASSERT(handle.is_for_cpu());
    ASSERT(!per_device_state.has_value());
    batch_norm_cpu_forward_kernel(
        /*attrs=*/attrs,
        /*input=*/input,
        /*gamma=*/gamma,
        /*beta=*/beta,
        /*output=*/output);
  }
}

void batch_norm_backward_kernel(
    device_stream_t const &stream,
    device_handle_t const &handle,
    std::optional<BatchNormPerDeviceState> const &per_device_state,
    BatchNormAttrs const &attrs,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &input_grad,
    GenericTensorAccessorR const &gamma,
    GenericTensorAccessorW const &gamma_grad,
    GenericTensorAccessorW const &beta_grad) {
  if (stream.is_gpu()) {
    batch_norm_gpu_backward_kernel(
        /*stream=*/stream.require_gpu(),
        /*handle=*/handle.require_for_gpu(),
        /*per_device_state=*/assert_unwrap(per_device_state),
        /*attrs=*/attrs,
        /*output=*/output,
        /*output_grad=*/output_grad,
        /*input=*/input,
        /*input_grad=*/input_grad,
        /*gamma=*/gamma,
        /*gamma_grad=*/gamma_grad,
        /*beta_grad=*/beta_grad);
  } else {
    ASSERT(stream.is_cpu());
    ASSERT(handle.is_for_cpu());
    ASSERT(!per_device_state.has_value());
    batch_norm_cpu_backward_kernel(
        /*attrs=*/attrs,
        /*output=*/output,
        /*output_grad=*/output_grad,
        /*input=*/input,
        /*input_grad=*/input_grad,
        /*gamma=*/gamma,
        /*gamma_grad=*/gamma_grad,
        /*beta_grad=*/beta_grad);
  }
}

void batch_norm_cleanup_kernel(
    DeviceType device_type,
    Allocator &allocator,
    std::optional<BatchNormPerDeviceState> &per_device_state) {
  if (device_type == DeviceType::GPU) {
    batch_norm_gpu_cleanup_kernel(allocator, per_device_state.value());
  } else {
    ASSERT(device_type == DeviceType::CPU);
    ASSERT(!per_device_state.has_value());
  }
}

} // namespace FlexFlow
