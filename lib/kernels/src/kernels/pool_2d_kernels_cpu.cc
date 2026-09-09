#include "kernels/pool_2d_kernels_cpu.h"
#include "utils/exception.h"

namespace FlexFlow {

void pool_2d_cpu_forward_kernel(Pool2DAttrs const &attrs,
                                GenericTensorAccessorR const &input,
                                GenericTensorAccessorW const &output) {
  NOT_IMPLEMENTED();
}

void pool_2d_cpu_backward_kernel(Pool2DAttrs const &attrs,
                                 GenericTensorAccessorR const &output,
                                 GenericTensorAccessorR const &output_grad,
                                 GenericTensorAccessorR const &input,
                                 GenericTensorAccessorW const &input_grad) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
