#include "kernels/pool_2d_kernels.h"
#include "kernels/pool_2d_kernels_cpu.h"
#include "kernels/pool_2d_kernels_gpu.h"
#include "utils/optional.h"

namespace FlexFlow {

std::optional<Pool2DPerDeviceState>
    pool_2d_init_kernel(DeviceType device_type,
                        Pool2DAttrs const &attrs,
                        TensorShape const &input_shape,
                        TensorShape const &output_shape) {
  if (device_type == DeviceType::GPU) {
    return pool_2d_gpu_init_kernel(
        /*attrs=*/attrs,
        /*input_shape=*/input_shape,
        /*output_shape=*/output_shape);
  } else {
    ASSERT(device_type == DeviceType::CPU);
    return std::nullopt;
  }
}

void pool_2d_forward_kernel(
    device_stream_t const &stream,
    device_handle_t const &handle,
    std::optional<Pool2DPerDeviceState> const &per_device_state,
    Pool2DAttrs const &attrs,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &output) {
  if (stream.is_gpu()) {
    pool_2d_gpu_forward_kernel(
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
    pool_2d_cpu_forward_kernel(
        /*attrs=*/attrs,
        /*input=*/input,
        /*output=*/output);
  }
}

void pool_2d_backward_kernel(
    device_stream_t const &stream,
    device_handle_t const &handle,
    std::optional<Pool2DPerDeviceState> const &per_device_state,
    Pool2DAttrs const &attrs,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &input_grad) {
  if (stream.is_gpu()) {
    pool_2d_gpu_backward_kernel(
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
    pool_2d_cpu_backward_kernel(
        /*attrs=*/attrs,
        /*output=*/output,
        /*output_grad=*/output_grad,
        /*input=*/input,
        /*input_grad=*/input_grad);
  }
}

void pool_2d_cleanup_kernel(
    DeviceType device_type,
    std::optional<Pool2DPerDeviceState> &per_device_state) {
  if (device_type == DeviceType::GPU) {
    pool_2d_gpu_cleanup_kernel(per_device_state.value());
  } else {
    ASSERT(device_type == DeviceType::CPU);
    ASSERT(!per_device_state.has_value());
  }
}

} // namespace FlexFlow
