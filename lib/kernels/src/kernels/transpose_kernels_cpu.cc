#include "kernels/transpose_kernels_cpu.h"
#include "utils/not_implemented.h"

namespace FlexFlow::Kernels::Transpose {

void cpu_forward_kernel(TransposeAttrs const &attrs,
                        GenericTensorAccessorR const &input,
                        GenericTensorAccessorW const &output) {
  NOT_IMPLEMENTED();
}

void cpu_backward_kernel(TransposeAttrs const &attrs,
                         GenericTensorAccessorR const &out_grad,
                         GenericTensorAccessorW const &in_grad) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow::Kernels::Transpose
