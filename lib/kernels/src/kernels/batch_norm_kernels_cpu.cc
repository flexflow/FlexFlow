#include "kernels/batch_norm_kernels_cpu.h"
#include "utils/exception.h"

namespace FlexFlow {

void batch_norm_cpu_forward_kernel(BatchNormAttrs const &attrs,
                                   GenericTensorAccessorR const &input,
                                   GenericTensorAccessorR const &gamma,
                                   GenericTensorAccessorR const &beta,
                                   GenericTensorAccessorW const &output) {
  NOT_IMPLEMENTED();
}

void batch_norm_cpu_backward_kernel(BatchNormAttrs const &attrs,
                                    GenericTensorAccessorR const &output,
                                    GenericTensorAccessorR const &output_grad,
                                    GenericTensorAccessorR const &input,
                                    GenericTensorAccessorW const &input_grad,
                                    GenericTensorAccessorR const &gamma,
                                    GenericTensorAccessorW const &gamma_grad,
                                    GenericTensorAccessorW const &beta_grad) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
