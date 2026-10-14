#include "kernels/concat_kernels_cpu.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

void concat_cpu_forward_kernel(
    GenericTensorAccessorW const &output,
    std::vector<GenericTensorAccessorR> const &inputs,
    ff_dim_t axis) {
  NOT_IMPLEMENTED();
}

void concat_cpu_backward_kernel(
    GenericTensorAccessorR const &output_grad,
    std::vector<GenericTensorAccessorW> const &input_grads,
    ff_dim_t axis) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
