#ifndef _FLEXFLOW_OPS_KERNELS_BATCH_MATMUL_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_BATCH_MATMUL_KERNELS_H

#include "kernels/device.h"
#include "kernels/ff_handle.h"
#include "kernels/per_device_op_state.h"

namespace FlexFlow {

class BatchMatmulPerDeviceState : public PerDeviceOpState {
public:
  BatchMatmulPerDeviceState(FFHandler handler);
  int a_seq_length_dim, b_seq_length_dim;
};

namespace Kernels {
namespace BatchMatmul {

void forward_kernel(ffStream_t stream,
                    BatchMatmulPerDeviceState const *,
                    float *o_ptr,
                    float const *a_ptr,
                    float const *b_ptr,
                    float const *c_ptr,
                    int m,
                    int n,
                    int k,
                    int batch,
                    int a_seq_length_dim = -1,
                    int b_seq_length_dim = -1,
                    int seq_length = -1);

void backward_kernel(ffStream_t stream,
                     BatchMatmulPerDeviceState const *,
                     float const *o_ptr,
                     float const *o_grad_ptr,
                     float const *a_ptr,
                     float *a_grad_ptr,
                     float const *b_ptr,
                     float *b_grad_ptr,
                     float *c_grad_ptr,
                     int m,
                     int n,
                     int k,
                     int batch);

} // namespace BatchMatmul
} // namespace Kernels
} // namespace FlexFlow

#endif
