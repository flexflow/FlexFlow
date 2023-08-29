#ifndef _FLEXFLOW_OPS_KERNELS_BATCH_MATMUL_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_BATCH_MATMUL_KERNELS_H

#include "kernels/device.h"
#include "kernels/ff_handle.h"

namespace FlexFlow {

struct BMMPerDeviceState {
  PerDeviceFFHandle handle;
  Allocator allocator;
  int a_seq_length_dim;
  req<int> b_seq_length_dim;
};

FF_VISITABLE_STRUCT_NO_EQ(
    BMMPerDeviceState, handle, allocator, a_seq_length_dim, b_seq_length_dim);

namespace Kernels {
namespace BatchMatmul {

BMMPerDeviceState init_kernel(PerDeviceFFHandle const &handle,
                              Allocator const &allocator,
                              int a_seq_length_dim,
                              int b_seq_length_dim);

void forward_kernel(ffStream_t stream,
                    BMMPerDeviceState const &meta,
                    float *output_ptr,
                    float const *a_ptr,
                    float const *b_ptr,
                    float const *c_ptr,
                    int m,
                    int n,
                    int k,
                    int batch,
                    int seq_length = -1);

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
