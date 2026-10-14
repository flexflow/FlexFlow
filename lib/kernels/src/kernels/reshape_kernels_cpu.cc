#include "kernels/reshape_kernels_cpu.h"
#include "utils/not_implemented.h"

namespace FlexFlow::Kernels::Reshape {

void cpu_forward_kernel(GenericTensorAccessorR const &input,
                        GenericTensorAccessorW const &output) {
  NOT_IMPLEMENTED();
}

void cpu_backward_kernel(GenericTensorAccessorR const &output,
                         GenericTensorAccessorW const &input) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow::Kernels::Reshape
