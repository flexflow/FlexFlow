#include "kernels/transpose_kernels_cpu.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

void transpose_cpu_forward_kernel(TransposeAttrs const &attrs,
                                  GenericTensorAccessorR const &input,
                                  GenericTensorAccessorW const &output) {
  NOT_IMPLEMENTED();
}

void transpose_cpu_backward_kernel(TransposeAttrs const &attrs,
                                   GenericTensorAccessorR const &out_grad,
                                   GenericTensorAccessorW const &in_grad) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
