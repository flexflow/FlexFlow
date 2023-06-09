#ifndef _FLEXFLOW_OPS_KERNELS_REDUCE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_REDUCE_KERNELS_H

#include "kernels/device.h"
#include "kernels/per_device_op_state.h"

namespace FlexFlow {

class ReducePerDeviceState : public PerDeviceOpState {
public:
  ReducePerDeviceState(FFHandler handler,
                       Reduce const *rd,
                       Legion::Domain const &input_domain);
  ~ReducePerDeviceState(void);
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnReduceTensorDescriptor_t reduceDesc;
#else
  miopenTensorDescriptor_t inputTensor, outputTensor;
  miopenReduceTensorDescriptor_t reduceDesc;
#endif
  OperatorType op_type;
  size_t reduction_size;
};

namespace Kernels {
namespace Reduce {
void forward_kernel_wrapper(ReducePerDeviceState const *m,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorW const &output);

void backward_kernel_wrapper(ReducePerDeviceState const *m,
                             GenericTensorAccessorR const &output_grad,
                             GenericTensorAccessorW const &input_grad);

namespace Internal {

void forward_kernel(ReducePerDeviceState const *m,
                    float const *input_ptr,
                    float *output_ptr,
                    ffStream_t stream);

void backward_kernel(ReducePerDeviceState const *m,
                     float const *output_grad_ptr,
                     float *input_grad_ptr,
                     ffStream_t stream);
} // namespace Internal
} // namespace Reduce
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_REDUCE_KERNELS_H
