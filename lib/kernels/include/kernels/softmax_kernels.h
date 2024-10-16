#ifndef _FLEXFLOW_OPS_KERNELS_SOFTMAX_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_SOFTMAX_KERNELS_H

#include "device.h"
#include "ff_handle.h"

namespace FlexFlow {

// Note(lambda): SoftmaxPerDeviceState may need add more elements
struct SoftmaxPerDeviceState {
  PerDeviceFFHandle handle;
  ffTensorDescriptor_t inputTensor;
  req<int> dim;
};

FF_VISITABLE_STRUCT(SoftmaxPerDeviceState, handle, inputTensor, dim);

namespace Kernels::Softmax {

SoftmaxPerDeviceState init_kernel(PerDeviceFFHandle const &handle,
                                  int dim,
                                  int input_n,
                                  int input_c,
                                  int input_h,
                                  int input_w);

void forward_kernel(ffStream_t stream,
                    SoftmaxPerDeviceState const &m,
                    float const *input_ptr,
                    float *output_ptr);

void backward_kernel(ffStream_t stream,
                     float *input_grad_ptr,
                     float const *output_grad_ptr,
                     size_t num_elements);

} // namespace Kernels::Softmax
} // namespace FlexFlow

#endif
