#include "kernels/flat_kernels_cpu.h"
#include "utils/not_implemented.h"

namespace FlexFlow::Kernels::Flat {

void cpu_forward_kernel(GenericTensorAccessorR const &input,
                        float *output_ptr) {
  NOT_IMPLEMENTED();
}

void cpu_backward_kernel(GenericTensorAccessorR const &input,
                         float const *output_grad_ptr,
                         float *input_grad_ptr) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow::Kernels::Flat
