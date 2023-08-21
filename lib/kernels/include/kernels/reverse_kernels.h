#ifndef _FLEXFLOW_OPS_KERNELS_REVERSE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_REVERSE_KERNELS_H

#include "kernels/device.h"

namespace FlexFlow {

namespace Kernels {
namespace Reverse {

void forward_kernel(ffStream_t stream,
                    float const *in_ptr,
                    float *out_ptr,
                    Legion::coord_t num_out_blks,
                    Legion::coord_t reverse_dim_size,
                    Legion::coord_t in_blk_size,
                    Legion::coord_t output_size);

void backward_kernel(ffStream_t stream,
                     float const *out_grad_ptr,
                     float *in_grad_ptr,
                     Legion::coord_t num_out_blks,
                     Legion::coord_t reverse_dim_size,
                     Legion::coord_t in_blk_size,
                     Legion::coord_t input_size);

} // namespace Reverse
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_REVERSE_KERNELS_H
