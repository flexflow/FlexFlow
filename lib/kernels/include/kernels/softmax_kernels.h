#ifndef _FLEXFLOW_OPS_KERNELS_SOFTMAX_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_SOFTMAX_KERNELS_H

#include "kernels/device.h"

namespace FlexFlow {

class SoftmaxPerDeviceState : public PerDeviceOpState {
public:
  SoftmaxPerDeviceState(FFHandler handle,
                        Softmax const *softmax,
                        Legion::Domain const &input_domain);
  ffTensorDescriptor_t inputTensor;
  bool profiling;
  int dim;
  char op_name[MAX_OPNAME];
};

namespace Kernels {
namespace Softmax {

void forward_kernel(ffStream_t stream,
                    SoftmaxPerDeviceState const *m,
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
