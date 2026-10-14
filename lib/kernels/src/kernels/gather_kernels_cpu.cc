#include "kernels/gather_kernels_cpu.h"
#include "utils/not_implemented.h"

namespace FlexFlow::Kernels::Gather {

void cpu_forward_kernel(GenericTensorAccessorR const &input,
                        GenericTensorAccessorR const &index,
                        GenericTensorAccessorW const &output) {
  NOT_IMPLEMENTED();
}

void cpu_backward_kernel(GenericTensorAccessorR const &output_grad,
                         GenericTensorAccessorR const &index,
                         GenericTensorAccessorW const &input_grad) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow::Kernels::Gather
