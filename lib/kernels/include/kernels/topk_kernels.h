#ifndef _FLEXFLOW_OPS_KERNELS_TOPK_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_TOPK_KERNELS_H

#include "kernels/device.h"
#include "kernels/per_device_op_state.h"

namespace FlexFlow {

class TopKPerDeviceState : public PerDeviceOpState {
public:
  TopKPerDeviceState(FFHandler handle);
  bool sorted;
};

namespace Kernels {
namespace TopK {
void forward_kernel_wrapper(TopKPerDeviceState const *m,
                                     float const *input_ptr,
                                     float *output_ptr,
                                     int *indices_ptr,
                                     size_t batch_size,
                                     int length,
                                     int k,
                                     bool sorted);
void backward_kernel_wrapper(TopKPerDeviceState const *m,
                                      float const *out_grad_ptr,
                                      int const *indices_ptr,
                                      float *in_grad_ptr,
                                      size_t batch_size,
                                      int length,
                                      int k);

namespace Internal {

void forward_kernel(TopKPerDeviceState const *m,
                             float const *input_ptr,
                             float *output_ptr,
                             int *indices_ptr,
                             size_t batch_size,
                             int length,
                             int k,
                             bool sorted,
                             ffStream_t stream);
void backward_kernel(TopKPerDeviceState const *m,
                              float const *out_grad_ptr,
                              int const *indices_ptr,
                              float *in_grad_ptr,
                              size_t batch_size,
                              int length,
                              int k,
                              ffStream_t stream);

} // namespace Internal
} // namespace TopK
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_TOPK_KERNELS_H
