#include "kernels/concat_kernels.h"
#include "kernels/concat_kernels_cpu.h"
#include "kernels/concat_kernels_gpu.h"

namespace FlexFlow {

void concat_forward_kernel(device_stream_t const &stream,
                           ConcatAttrs const &attrs,
                           std::vector<GenericTensorAccessorR> const &inputs,
                           GenericTensorAccessorW const &output) {
  if (stream.is_gpu()) {
    concat_gpu_forward_kernel(
        /*stream=*/stream.require_gpu(),
        /*attrs=*/attrs,
        /*inputs=*/inputs,
        /*output=*/output);
  } else {
    ASSERT(stream.is_cpu());
    concat_cpu_forward_kernel(
        /*attrs=*/attrs,
        /*inputs=*/inputs,
        /*output=*/output);
  }
}

void concat_backward_kernel(
    device_stream_t const &stream,
    ConcatAttrs const &attrs,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad,
    std::vector<GenericTensorAccessorR> const &inputs,
    std::vector<GenericTensorAccessorW> const &input_grads) {
  if (stream.is_gpu()) {
    concat_gpu_backward_kernel(
        /*stream=*/stream.require_gpu(),
        /*attrs=*/attrs,
        /*output=*/output,
        /*output_grad=*/output_grad,
        /*inputs=*/inputs,
        /*input_grads=*/input_grads);
  } else {
    ASSERT(stream.is_cpu());
    concat_cpu_backward_kernel(
        /*attrs=*/attrs,
        /*output=*/output,
        /*output_grad=*/output_grad,
        /*inputs=*/inputs,
        /*input_grads=*/input_grads);
  }
}

} // namespace FlexFlow
