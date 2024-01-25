#ifndef _FLEXFLOW_OPS_KERNELS_ELEMENT_UNARY_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_ELEMENT_UNARY_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device.h"
#include "kernels/ff_handle.h"
#include "op-attrs/ops/element_unary.h"
#include <cstddef>

namespace FlexFlow {

struct ElementUnaryPerDeviceState {
  ffTensorDescriptor_t inputTensor, outputTensor;
  ffActivationDescriptor_t actiDesc;
};

FF_VISITABLE_STRUCT_NO_EQ(ElementUnaryPerDeviceState,
                          inputTensor,
                          outputTensor,
                          actiDesc);

namespace Kernels {
namespace ElementUnary {

ElementUnaryPerDeviceState init_kernel(ArrayShape const &input_shape,
                                       ArrayShape const &output_shape,
                                       ElementUnaryAttrs const &attrs);

void forward_kernel(ffStream_t stream,
                    ElementUnaryPerDeviceState const &device_state,
                    ElementUnaryAttrs const &attrs,
                    PerDeviceFFHandle &handle,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output);

void backward_kernel(ffStream_t stream,
                     ElementUnaryPerDeviceState const &device_state,
                     ElementUnaryAttrs const &attrs,
                     PerDeviceFFHandle &handle,
                     GenericTensorAccessorR const &input,
                     GenericTensorAccessorW const &input_grad,
                     GenericTensorAccessorR const &output,
                     GenericTensorAccessorR const &output_grad);

} // namespace ElementUnary
} // namespace Kernels
} // namespace FlexFlow

#endif
