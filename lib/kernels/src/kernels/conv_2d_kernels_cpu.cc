#include "kernels/conv_2d_kernels_cpu.h"
#include "utils/exception.h"

namespace FlexFlow {

void conv_2d_cpu_forward_kernel(
    Conv2DAttrs const &attrs,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorR const &filter,
    std::optional<GenericTensorAccessorR> const &bias,
    GenericTensorAccessorW const &output) {
  NOT_IMPLEMENTED();
}

void conv_2d_cpu_backward_kernel(
    Conv2DAttrs const &attrs,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &input_grad,
    GenericTensorAccessorR const &filter,
    GenericTensorAccessorW const &filter_grad,
    std::optional<GenericTensorAccessorW> const &bias_grad) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
