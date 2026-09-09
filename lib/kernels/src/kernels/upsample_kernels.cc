#include "kernels/upsample_kernels.h"
#include "kernels/upsample_kernels_cpu.h"
#include "kernels/upsample_kernels_gpu.h"

namespace FlexFlow {

void upsample_forward_kernel(device_stream_t const &stream,
                             UpsampleAttrs const &attrs,
                             GenericTensorAccessorR const &input,
                             GenericTensorAccessorW const &output) {
  if (stream.is_gpu()) {
    upsample_gpu_forward_kernel(
        /*stream=*/stream.require_gpu(),
        /*attrs=*/attrs,
        /*input=*/input,
        /*output=*/output);
  } else {
    ASSERT(stream.is_cpu());

    upsample_cpu_forward_kernel(
        /*attrs=*/attrs,
        /*input=*/input,
        /*output=*/output);
  }
}

void upsample_backward_kernel(device_stream_t const &stream,
                              UpsampleAttrs const &attrs,
                              GenericTensorAccessorR const &output,
                              GenericTensorAccessorR const &output_grad,
                              GenericTensorAccessorR const &input,
                              GenericTensorAccessorW const &input_grad) {
  if (stream.is_gpu()) {
    upsample_gpu_backward_kernel(
        /*stream=*/stream.require_gpu(),
        /*attrs=*/attrs,
        /*output=*/output,
        /*output_grad=*/output_grad,
        /*input=*/input,
        /*input_grad=*/input_grad);
  } else {
    ASSERT(stream.is_cpu());

    upsample_cpu_backward_kernel(
        /*attrs=*/attrs,
        /*output=*/output,
        /*output_grad=*/output_grad,
        /*input=*/input,
        /*input_grad=*/input_grad);
  }
}

} // namespace FlexFlow
