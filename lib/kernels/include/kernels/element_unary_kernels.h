#ifndef _FLEXFLOW_OPS_KERNELS_ELEMENT_UNARY_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_ELEMENT_UNARY_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device.h"
#include "kernels/ff_handle.h"
#include "op-attrs/ops/element_unary.h"
#include <cstddef>

namespace FlexFlow {

struct ElementUnaryPerDeviceState {
  PerDeviceFFHandle handle;
  ffTensorDescriptor_t inputTensor, outputTensor;
  ffActivationDescriptor_t actiDesc;

  OperatorType op_type;
  DataType data_type;
  float scalar;
};

FF_VISITABLE_STRUCT_NO_EQ(ElementUnaryPerDeviceState,
                          handle,
                          inputTensor,
                          outputTensor,
                          actiDesc,
                          op_type,
                          data_type,
                          scalar);

namespace Kernels {
namespace ElementUnary {

ElementUnaryPerDeviceState init_kernel(PerDeviceFFHandle const &handle,
                                       ArrayShape const &input_shape,
                                       ArrayShape const &output_shape,
                                       DataType data_type);

void forward_kernel(ffStream_t stream,
                    ElementUnaryPerDeviceState const &device_state,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output);

void backward_kernel(ffStream_t stream,
                     ElementUnaryPerDeviceState const &device_state,
                     GenericTensorAccessorR const &input,
                     GenericTensorAccessorW const &input_grad,
                     GenericTensorAccessorR const &output,
                     GenericTensorAccessorR const &output_grad);

} // namespace ElementUnary
} // namespace Kernels
} // namespace FlexFlow

#endif
