#ifndef _FLEXFLOW_OPS_KERNELS_SPLIT_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_SPLIT_KERNELS_H

#include "device.h"

namespace FlexFlow {

using legion_coord_t = long long;

namespace Kernels {
namespace Split {
void forward_kernel(ffStream_t stream,
                    float **out_ptrs,
                    float const *in_ptr,
                    legion_coord_t const *out_blk_sizes,
                    legion_coord_t in_blk_size,
                    legion_coord_t num_blks,
                    int numOutputs);
void backward_kernel(ffStream_t stream,
                     float *in_grad_ptr,
                     float const **out_grad_ptr,
                     legion_coord_t const *out_blk_sizes,
                     legion_coord_t in_blk_size,
                     legion_coord_t num_blks,
                     int numOutputs);

} // namespace Split
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_SPLIT_KERNELS_H
