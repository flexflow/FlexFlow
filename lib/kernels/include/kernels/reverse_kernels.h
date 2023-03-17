#ifndef _FLEXFLOW_OPS_KERNELS_REVERSE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_REVERSE_KERNELS_H

#include "runtime/device.h"
#include "runtime/fftype.h"
#include "runtime/op_meta.h"

namespace FlexFlow {

namespace Kernels {
namespace Reverse {
void forward_kernel_wrapper(float const *in_ptr,
                                     float *out_ptr,
                                     Legion::coord_t num_out_blks,
                                     Legion::coord_t reverse_dim_size,
                                     Legion::coord_t in_blk_size,
                                     Legion::coord_t output_size);

void backward_kernel_wrapper(float const *out_grad_ptr,
                                      float *in_grad_ptr,
                                      Legion::coord_t num_out_blks,
                                      Legion::coord_t reverse_dim_size,
                                      Legion::coord_t in_blk_size,
                                      Legion::coord_t input_size);

namespace Internal {

void forward_kernel(float const *in_ptr,
                             float *out_ptr,
                             Legion::coord_t num_out_blks,
                             Legion::coord_t reverse_dim_size,
                             Legion::coord_t in_blk_size,
                             Legion::coord_t output_size,
                             ffStream_t stream);

void backward_kernel(float const *out_grad_ptr,
                              float *in_grad_ptr,
                              Legion::coord_t num_out_blks,
                              Legion::coord_t reverse_dim_size,
                              Legion::coord_t in_blk_size,
                              Legion::coord_t input_size,
                              ffStream_t stream);
} // namespace Internal
} // namespace Reverse
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_REVERSE_KERNELS_H
