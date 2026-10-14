#include "kernels/conv_2d_kernels_cpu.h"
#include "utils/not_implemented.h"

namespace FlexFlow::Kernels::Conv2D {

void cpu_forward_kernel(float const *input_ptr,
                        float *output_ptr,
                        float const *filter_ptr,
                        float const *bias_ptr,
                        std::optional<Activation> const &activation) {
  NOT_IMPLEMENTED();
}

void cpu_backward_kernel(float const *output_ptr,
                         float *output_grad_ptr,
                         float const *input_ptr,
                         float *input_grad_ptr,
                         float const *filter_ptr,
                         float *filter_grad_ptr,
                         float *bias_grad_ptr,
                         std::optional<Activation> const &activation) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow::Kernels::Conv2D
