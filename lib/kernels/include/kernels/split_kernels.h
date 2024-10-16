#ifndef _FLEXFLOW_OPS_KERNELS_SPLIT_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_SPLIT_KERNELS_H

#include "device.h"

namespace FlexFlow::Kernels::Split {
void forward_kernel(ffStream_t stream,
                    float **out_ptrs,
                    float const *in_ptr,
                    coord_t const *out_blk_sizes,
                    coord_t in_blk_size,
                    coord_t num_blks,
                    int numOutputs);
void backward_kernel(ffStream_t stream,
                     float *in_grad_ptr,
                     float const **out_grad_ptr,
                     coord_t const *out_blk_sizes,
                     coord_t in_blk_size,
                     coord_t num_blks,
                     int numOutputs);

} // namespace FlexFlow::Kernels::Split

#endif // _FLEXFLOW_OPS_KERNELS_SPLIT_KERNELS_H
