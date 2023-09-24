#ifndef _FLEXFLOW_OPS_KERNELS_REDUCE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_REDUCE_KERNELS_H

#include "kernels/device.h"

namespace FlexFlow {

struct ReducePerDeviceState {
  PerDeviceFFHandle handle;
  ffTensorDescriptor_t inputTensor;
  ffTensorDescriptor_t outputTensor;
  ffReduceTensorDescriptor_t reduceDesc;
  OperatorType op_type;
  size_t reduction_size;
};

FF_VISITABLE_STRUCT(ReducePerDeviceState,
                          handle,
                          inputTensor,
                          outputTensor,
                          reduceDesc,
                          op_type,
                          reduction_size);

namespace Kernels {
namespace Reduce {

ReducePerDeviceState init_kernel(PerDeviceFFhandle const &,
                                 OperatorType const &,
                                 size_t const &,
                                 ArrayShape input_shape,
                                 ArrayShape output_shape);

void forward_kernel_wrapper(ReducePerDeviceState const &m,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorW const &output);

void backward_kernel_wrapper(ReducePerDeviceState const &m,
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
