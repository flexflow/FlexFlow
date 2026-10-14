#include "kernels/attention_kernels_cpu.h"
#include "utils/not_implemented.h"

namespace FlexFlow::Kernels::MultiHeadAttention {

void cpu_forward_kernel(float const *query_ptr,
                        float const *key_ptr,
                        float const *value_ptr,
                        float const *weight_ptr,
                        float *output_ptr) {
  NOT_IMPLEMENTED();
}

void cpu_backward_kernel(float const *query_ptr,
                         float *query_grad_ptr,
                         float const *key_ptr,
                         float *key_grad_ptr,
                         float const *value_ptr,
                         float *value_grad_ptr,
                         float const *weight_ptr,
                         float *weight_grad_ptr,
                         float const *output_grad_ptr) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow::Kernels::MultiHeadAttention
