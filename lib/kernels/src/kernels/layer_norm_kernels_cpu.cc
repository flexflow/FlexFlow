#include "kernels/layer_norm_kernels_cpu.h"
#include "utils/not_implemented.h"

namespace FlexFlow::Kernels::LayerNorm {

void cpu_forward_kernel(GenericTensorAccessorR const &input,
                        GenericTensorAccessorW const &output,
                        GenericTensorAccessorW const &gamma,
                        GenericTensorAccessorW const &beta) {
  NOT_IMPLEMENTED();
}

void cpu_backward_kernel(GenericTensorAccessorR const &output_grad,
                         GenericTensorAccessorR const &input,
                         GenericTensorAccessorW const &input_grad,
                         GenericTensorAccessorR const &gamma,
                         GenericTensorAccessorW const &gamma_grad,
                         GenericTensorAccessorW const &beta_grad) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow::Kernels::LayerNorm
