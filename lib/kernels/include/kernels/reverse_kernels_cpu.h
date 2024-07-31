#ifndef _FLEXFLOW_OPS_KERNELS_REVERSE_KERNELS_CPU_H
#define _FLEXFLOW_OPS_KERNELS_REVERSE_KERNELS_CPU_H

#include "device.h"

namespace FlexFlow {
namespace Kernels {
namespace Reverse {

void cpu_forward_kernel(float const *in_ptr,
                        float *out_ptr,
                        coord_t num_out_blks,
                        coord_t reverse_dim_size,
                        coord_t in_blk_size,
                        coord_t output_size);

void cpu_backward_kernel(float const *out_grad_ptr,
                         float *in_grad_ptr,
                         coord_t num_out_blks,
                         coord_t reverse_dim_size,
                         coord_t in_blk_size,
                         coord_t input_size);
} // namespace Reverse
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_REVERSE_KERNELS_CPU_H
