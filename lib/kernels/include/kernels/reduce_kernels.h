#ifndef _FLEXFLOW_OPS_KERNELS_REDUCE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_REDUCE_KERNELS_H

#include "array_shape.h"
#include "device.h"
#include "ff_handle.h"
#include "op-attrs/op.h"

namespace FlexFlow {

struct ReducePerDeviceState {
  PerDeviceFFHandle handle;
  ffTensorDescriptor_t inputTensor;
  ffTensorDescriptor_t outputTensor;
  ffReduceTensorDescriptor_t reduceDesc;
  OperatorType op_type;
  req<size_t> reduction_size;
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

ReducePerDeviceState init_kernel(PerDeviceFFHandle const &,
                                 OperatorType const &,
                                 size_t const &,
                                 ArrayShape input_shape,
                                 ArrayShape output_shape);

void forward_kernel(ffStream_t stream,
                    ReducePerDeviceState const &m,
                    float const *input_ptr,
                    float *output_ptr);

void backward_kernel(ffStream_t stream,
                     ReducePerDeviceState const &m,
                     float const *output_grad_ptr,
                     float *input_grad_ptr);
} // namespace Reduce
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_REDUCE_KERNELS_H
