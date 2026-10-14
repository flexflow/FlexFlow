#include "kernels/topk_kernels_cpu.h"
#include "utils/not_implemented.h"

namespace FlexFlow::Kernels::TopK {

void cpu_forward_kernel(float const *input_ptr,
                        float *output_ptr,
                        int *indices_ptr,
                        size_t batch_size,
                        int length,
                        int k,
                        bool sorted) {
  NOT_IMPLEMENTED();
}

void cpu_backward_kernel(float const *out_grad_ptr,
                         int const *indices_ptr,
                         float *in_grad_ptr,
                         size_t batch_size,
                         int length,
                         int k) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow::Kernels::TopK
