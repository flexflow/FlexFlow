#include "kernels/split_kernels_cpu.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

void split_cpu_forward_kernel(
    SplitAttrs const &attrs,
    GenericTensorAccessorR const &input,
    std::vector<GenericTensorAccessorW> const &outputs) {
  NOT_IMPLEMENTED();
}

void split_cpu_backward_kernel(
    SplitAttrs const &attrs,
    std::vector<GenericTensorAccessorR> const &output_grads,
    GenericTensorAccessorW const &input_grad) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
