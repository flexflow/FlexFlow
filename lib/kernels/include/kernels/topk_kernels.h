#ifndef _FLEXFLOW_OPS_KERNELS_TOPK_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_TOPK_KERNELS_H

#include "device.h"
#include "kernels/allocation.h"

namespace FlexFlow {

struct TopKPerDeviceState {
  req<bool> sorted; // Note: Does TopK needs a PerDeviceFFHandle handle?
};

FF_VISITABLE_STRUCT(TopKPerDeviceState, sorted);

namespace Kernels::TopK {

TopKPerDeviceState init_kernel(bool sorted);

void forward_kernel(ffStream_t stream,
                    TopKPerDeviceState const &m,
                    float const *input_ptr,
                    float *output_ptr,
                    int *indices_ptr,
                    size_t batch_size,
                    int length,
                    int k,
                    bool sorted);
void backward_kernel(ffStream_t stream,
                     TopKPerDeviceState const &m,
                     float const *out_grad_ptr,
                     int const *indices_ptr,
                     float *in_grad_ptr,
                     size_t batch_size,
                     int length,
                     int k);

} // namespace Kernels::TopK
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_TOPK_KERNELS_H
