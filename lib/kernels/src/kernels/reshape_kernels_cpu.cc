#include "kernels/reshape_kernels_cpu.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

void reshape_cpu_forward_kernel(GenericTensorAccessorR const &input,
                                GenericTensorAccessorW const &output) {
  NOT_IMPLEMENTED();
}

void reshape_cpu_backward_kernel(GenericTensorAccessorR const &output_grad,
                                 GenericTensorAccessorW const &input_grad) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
