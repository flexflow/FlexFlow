#ifndef _FLEXFLOW_OPS_KERNELS_SOFTMAX_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_SOFTMAX_KERNELS_H

#include "ff_handle.h"
#include "kernels/device.h"
#include "kernels/ff_handler.h"

namespace FlexFlow {

// Note(lambda): SoftmaxPerDeviceState may need add more elements
struct SoftmaxPerDeviceState {
  PerDeviceFFHandle handle;
  ffTensorDescriptor_t inputTensor;
};

FF_VISITABLE_STRUCT(SoftmaxPerDeviceState, handle, inputTensor);

namespace Kernels {
namespace Softmax {

SoftmaxPerDeviceState init_kernel(PerDeviceFFHandle const &);

void forward_kernel(ffStream_t stream,
                    SoftmaxPerDeviceState const &m,
                    float const *input_ptr,
                    float *output_ptr);
void backward_kernel(ffStream_t stream,
                     float *input_grad_ptr,
                     float const *output_grad_ptr,
                     size_t num_elements);

} // namespace Softmax
} // namespace Kernels
} // namespace FlexFlow

#endif
