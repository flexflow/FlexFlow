#include "kernels/dropout_kernels_cpu.h"
#include "utils/not_implemented.h"

namespace FlexFlow::Kernels::Dropout {

void cpu_forward_kernel(float const *input_ptr, float *output_ptr) {
  NOT_IMPLEMENTED();
}

void cpu_backward_kernel(float const *output_grad_ptr, float *input_grad_ptr) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow::Kernels::Dropout
