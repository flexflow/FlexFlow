#include "kernels/reduce_kernels.h"
#include "kernels/reduce_kernels_cpu.h"
#include "kernels/reduce_kernels_gpu.h"
#include <libassert/assert.hpp>

namespace FlexFlow::Kernels::Reduce {

std::optional<ReducePerDeviceState>
    init_kernel(DeviceType device_type,
                device_handle_t const &handle,
                ReduceOp const &reduce_op,
                size_t const &reduction_size,
                TensorShape const &input_shape,
                TensorShape const &output_shape) {
  if (device_type == DeviceType::GPU) {
    return gpu_init_kernel(/*handle=*/handle.require_for_gpu(),
                           /*reduce_op=*/reduce_op,
                           /*reduction_size=*/reduction_size,
                           /*input_shape=*/input_shape,
                           /*output_shape=*/output_shape);
  } else {
    ASSERT(device_type == DeviceType::CPU);
    ASSERT(handle.is_for_cpu());
    return std::nullopt;
  }
}

void forward_kernel(device_stream_t const &stream,
                    std::optional<ReducePerDeviceState> const &per_device_state,
                    float const *input_ptr,
                    float *output_ptr) {
  if (stream.is_gpu()) {
    gpu_forward_kernel(/*stream=*/stream.require_gpu(),
                       /*per_device_state=*/per_device_state.value(),
                       /*input_ptr=*/input_ptr,
                       /*output_ptr=*/output_ptr);
  } else {
    ASSERT(stream.is_cpu());
    ASSERT(per_device_state == std::nullopt);
    cpu_forward_kernel(/*input_ptr=*/input_ptr,
                       /*output_ptr=*/output_ptr);
  }
}

void backward_kernel(
    device_stream_t const &stream,
    std::optional<ReducePerDeviceState> const &per_device_state,
    float const *output_grad_ptr,
    float *input_grad_ptr) {
  if (stream.is_gpu()) {
    gpu_backward_kernel(/*stream=*/stream.require_gpu(),
                        /*per_device_state=*/per_device_state.value(),
                        /*output_grad_ptr=*/output_grad_ptr,
                        /*input_grad_ptr=*/input_grad_ptr);
  } else {
    ASSERT(stream.is_cpu());
    ASSERT(per_device_state == std::nullopt);
    cpu_backward_kernel(/*output_grad_ptr=*/output_grad_ptr,
                        /*input_grad_ptr=*/input_grad_ptr);
  }
}

} // namespace FlexFlow::Kernels::Reduce
