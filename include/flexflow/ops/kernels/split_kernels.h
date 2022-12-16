#ifndef _FLEXFLOW_OPS_KERNELS_SPLIT_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_SPLIT_KERNELS_H

#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"

namespace FlexFlow {

namespace Kernels {
namespace Split {
void forward_kernel_wrapper(float **out_ptrs,
                            float const *in_ptr,
                            Legion::coord_t const *out_blk_sizes,
                            Legion::coord_t in_blk_size,
                            Legion::coord_t num_blks,
                            int numOutputs);

void backward_kernel_wrapper(float *in_grad_ptr,
                             float const **out_grad_ptr,
                             Legion::coord_t const *out_blk_sizes,
                             Legion::coord_t in_blk_size,
                             Legion::coord_t num_blks,
                             int numOutputs);

namespace Internal {
void forward_kernel(float **out_ptrs,
                    float const *in_ptr,
                    Legion::coord_t const *out_blk_sizes,
                    Legion::coord_t in_blk_size,
                    Legion::coord_t num_blks,
                    int numOutputs,
                    ffStream_t stream);
void backward_kernel(float *in_grad_ptr,
                     float const **out_grad_ptr,
                     Legion::coord_t const *out_blk_sizes,
                     Legion::coord_t in_blk_size,
                     Legion::coord_t num_blks,
                     int numOutputs,
                     ffStream_t stream);
} // namespace Internal
} // namespace Split
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_SPLIT_KERNELS_H
