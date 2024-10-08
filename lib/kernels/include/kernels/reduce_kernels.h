#ifndef _FLEXFLOW_OPS_KERNELS_REDUCE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_REDUCE_KERNELS_H

#include "array_shape.h"
#include "device.h"
#include "ff_handle.h"
#include "op-attrs/operator_type.dtg.h"

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

namespace Kernels::Reduce {

ReducePerDeviceState init_kernel(PerDeviceFFHandle const &,
                                 OperatorType const &,
                                 size_t const &,
                                 ArrayShape const &input_shape,
                                 ArrayShape const &output_shape);

void forward_kernel(ffStream_t stream,
                    ReducePerDeviceState const &m,
                    float const *input_ptr,
                    float *output_ptr);

void backward_kernel(ffStream_t stream,
                     ReducePerDeviceState const &m,
                     float const *output_grad_ptr,
                     float *input_grad_ptr);
} // namespace Kernels::Reduce
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_REDUCE_KERNELS_H
