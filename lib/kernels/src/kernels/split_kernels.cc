#include "kernels/split_kernels.h"
#include "kernels/split_kernels_cpu.h"
#include "kernels/split_kernels_gpu.h"

namespace FlexFlow {

void split_forward_kernel(device_stream_t const &stream,
                          SplitAttrs const &attrs,
                          GenericTensorAccessorR const &input,
                          std::vector<GenericTensorAccessorW> const &outputs) {
  if (stream.is_gpu()) {
    split_gpu_forward_kernel(
        /*stream=*/stream.require_gpu(),
        /*attrs=*/attrs,
        /*input=*/input,
        /*outputs=*/outputs);
  } else {
    ASSERT(stream.is_cpu());
    split_cpu_forward_kernel(
        /*attrs=*/attrs,
        /*input=*/input,
        /*outputs=*/outputs);
  }
}

void split_backward_kernel(
    device_stream_t const &stream,
    SplitAttrs const &attrs,
    std::vector<GenericTensorAccessorR> const &outputs,
    std::vector<GenericTensorAccessorR> const &output_grads,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &input_grad) {
  if (stream.is_gpu()) {
    split_gpu_backward_kernel(
        /*stream=*/stream.require_gpu(),
        /*attrs=*/attrs,
        /*outputs=*/outputs,
        /*output_grads=*/output_grads,
        /*input=*/input,
        /*input_grad=*/input_grad);
  } else {
    ASSERT(stream.is_cpu());
    split_cpu_backward_kernel(
        /*attrs=*/attrs,
        /*outputs=*/outputs,
        /*output_grads=*/output_grads,
        /*input=*/input,
        /*input_grad=*/input_grad);
  }
}

} // namespace FlexFlow
