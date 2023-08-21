#ifndef _FLEXFLOW_OPS_KERNELS_SPLIT_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_SPLIT_KERNELS_H

#include "kernels/device.h"

namespace FlexFlow {

namespace Kernels {
namespace Split {
void forward_kernel(ffStream_t stream,
                    float **out_ptrs,
                    float const *in_ptr,
                    Legion::coord_t const *out_blk_sizes,
                    Legion::coord_t in_blk_size,
                    Legion::coord_t num_blks,
                    int numOutputs);
void backward_kernel(ffStream_t stream,
                     float *in_grad_ptr,
                     float const **out_grad_ptr,
                     Legion::coord_t const *out_blk_sizes,
                     Legion::coord_t in_blk_size,
                     Legion::coord_t num_blks,
                     int numOutputs);

} // namespace Split
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_SPLIT_KERNELS_H
