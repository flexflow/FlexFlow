#include "kernels/pool_2d_kernels_cpu.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

void pool2d_cpu_forward_kernel(Pool2DAttrs const &attrs,
                               GenericTensorAccessorR const &input,
                               GenericTensorAccessorW const &output) {
  NOT_IMPLEMENTED();
}

void pool2d_cpu_backward_kernel(Pool2DAttrs const &attrs,
                                GenericTensorAccessorR const &output_grad,
                                GenericTensorAccessorW const &input_grad) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
