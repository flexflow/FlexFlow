#include "kernels/split_kernels_cpu.h"
#include "utils/not_implemented.h"

namespace FlexFlow::Kernels::Split {

void cpu_forward_kernel(float **out_ptrs,
                        float const *in_ptr,
                        int const *out_blk_sizes,
                        int in_blk_size,
                        int num_blks,
                        int numOutputs) {
  NOT_IMPLEMENTED();
}

void cpu_backward_kernel(float *in_grad_ptr,
                         float const **out_grad_ptr,
                         int const *out_blk_sizes,
                         int in_blk_size,
                         int num_blks,
                         int numOutputs) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow::Kernels::Split
