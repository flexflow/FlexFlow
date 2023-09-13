#ifndef _FLEXFLOW_OPS_KERNELS_BATCH_MATMUL_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_BATCH_MATMUL_KERNELS_H

#include "kernels/allocation.h"
#include "kernels/device.h"
#include "kernels/ff_handle.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct BMMPerDeviceState {
  PerDeviceFFHandle handle;
};

FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(BMMPerDeviceState, handle);

namespace Kernels {
namespace BatchMatmul {

BMMPerDeviceState init_kernel(PerDeviceFFHandle const &handle,
                              Allocator const &allocator);

void forward_kernel(ffStream_t stream,
                    BMMPerDeviceState const &meta,
                    float *output_ptr,
                    float const *a_input_ptr,
                    float const *b_input_ptr,
                    int m,
                    int n,
                    int k,
                    int batch,
                    int seq_length,
                    int a_seq_length_dim,
                    int b_seq_length_dim);

void backward_kernel(ffStream_t stream,
                     BMMPerDeviceState const &meta,
                     float const *o_ptr,
                     float const *o_grad_ptr,
                     float const *a_ptr,
                     float *a_grad_ptr,
                     float const *b_ptr,
                     float *b_grad_ptr,
                     int m,
                     int n,
                     int k,
                     int batch);

} // namespace BatchMatmul
} // namespace Kernels
} // namespace FlexFlow

#endif
