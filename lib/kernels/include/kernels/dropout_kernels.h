#ifndef _FLEXFLOW_OPS_KERNELS_DROPOUT_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_DROPOUT_KERNELS_H

#include "kernels/device.h"
#include "kernels/per_device_op_state.h"
#include <cstddef>

namespace FlexFlow {

class DropoutPerDeviceState : public PerDeviceOpState {
public:
  DropoutPerDeviceState(FFHandler handler,
                        float rate,
                        unsigned long long seed,
                        bool profiling,
                        Legion::Memory gpu_mem,
                        Legion::Domain const &output_domain);
  ~DropoutPerDeviceState(void);
  Realm::RegionInstance reserveInst;
  ffTensorDescriptor_t inputTensor, outputTensor;
  ffDropoutDescriptor_t dropoutDesc;
  void *reserveSpace, *dropoutStates;
  size_t reserveSpaceSize, dropoutStateSize;
};

namespace Kernels {
namespace Dropout {
void forward_kernel(ffStream_t stream,
                    DropoutPerDeviceState *m,
                    float const *input_ptr,
                    float *output_ptr);
void backward_kernel(ffStream_t stream,
                     DropoutPerDeviceState *m,
                     float const *output_grad_ptr,
                     float *input_grad_ptr);

} // namespace Dropout
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_DROPOUT_KERNELS_H
