#ifndef _FLEXFLOW_OPS_KERNELS_REVERSE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_REVERSE_KERNELS_H

#include "device.h"

namespace FlexFlow::Kernels::Reverse {

void forward_kernel(ffStream_t stream,
                    float const *in_ptr,
                    float *out_ptr,
                    coord_t num_out_blks,
                    coord_t reverse_dim_size,
                    coord_t in_blk_size,
                    coord_t output_size);

void backward_kernel(ffStream_t stream,
                     float const *out_grad_ptr,
                     float *in_grad_ptr,
                     coord_t num_out_blks,
                     coord_t reverse_dim_size,
                     coord_t in_blk_size,
                     coord_t input_size);

} // namespace FlexFlow::Kernels::Reverse

#endif // _FLEXFLOW_OPS_KERNELS_REVERSE_KERNELS_H
