#include "kernels/reshape_kernels.h"
#include "kernels/reshape_kernels_cpu.h"
#include "kernels/reshape_kernels_gpu.h"

namespace FlexFlow {

void reshape_forward_kernel(device_stream_t const &stream,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorW const &output) {
  if (stream.is_gpu()) {
    reshape_gpu_forward_kernel(
        /*stream=*/stream.require_gpu(),
        /*input=*/input,
        /*output=*/output);
  } else {
    ASSERT(stream.is_cpu());
    reshape_cpu_forward_kernel(
        /*input=*/input,
        /*output=*/output);
  }
}

void reshape_backward_kernel(device_stream_t const &stream,
                             GenericTensorAccessorR const &output,
                             GenericTensorAccessorW const &input) {
  if (stream.is_gpu()) {
    reshape_gpu_backward_kernel(
        /*stream=*/stream.require_gpu(),
        /*output=*/output,
        /*input=*/input);
  } else {
    ASSERT(stream.is_cpu());
    reshape_cpu_backward_kernel(
        /*output=*/output,
        /*input=*/input);
  }
}

} // namespace FlexFlow
