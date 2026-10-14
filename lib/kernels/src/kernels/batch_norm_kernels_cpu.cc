#include "kernels/batch_norm_kernels_cpu.h"
#include "utils/not_implemented.h"

namespace FlexFlow::Kernels::BatchNorm {

void cpu_forward_kernel(BatchNormPerDeviceState const &per_device_state,
                        float const *input_ptr,
                        float *output_ptr,
                        float const *scale_ptr,
                        float const *bias_ptr) {
  NOT_IMPLEMENTED();
}

void cpu_backward_kernel(BatchNormPerDeviceState const &per_device_state,
                         float const *output_ptr,
                         float *output_grad_ptr,
                         float const *input_ptr,
                         float *input_grad_ptr,
                         float const *scale_ptr,
                         float *scale_grad_ptr,
                         float *bias_grad_ptr,
                         size_t numElements) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow::Kernels::BatchNorm
