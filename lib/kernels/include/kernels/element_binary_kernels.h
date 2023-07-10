#ifndef _FLEXFLOW_OPS_KERNELS_ELEMENT_BINARY_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_ELEMENT_BINARY_KERNELS_H

#include "kernels/array_shape.h"
#include "kernels/device.h"
#include "kernels/per_device_op_state.h"

namespace FlexFlow {

class ElementBinaryPerDeviceState : public PerDeviceOpState {
public:
  ElementBinaryPerDeviceState(FFHandler handle);
  ffTensorDescriptor_t input1Tensor, input2Tensor, outputTensor;
  ffOpTensorDescriptor_t opDesc;
  ffReduceTensorDescriptor_t reduceAddDesc;
  OperatorType op_type;
  bool inplace_a, has_same_operands;
  bool broadcast_input1, broadcast_input2;
  char op_name[MAX_OPNAME];
};

namespace Kernels {
namespace ElementBinary {

void init_kernel(ElementBinaryPerDeviceState *m,
                 ArrayShape const &input1_domain,
                 ArrayShape const &input2_domain,
                 ArrayShape const &output_domain);

void forward_kernel(ffStream_t stream,
                    ElementBinaryPerDeviceState const *m,
                    float const *in1_ptr,
                    float const *in2_ptr,
                    float *out_ptr);
void backward_kernel(ffStream_t stream,
                     ElementBinaryPerDeviceState const *m,
                     float const *out_grad_ptr,
                     float const *in1_ptr,
                     float const *in2_ptr,
                     float *in1_grad_ptr,
                     float *in2_grad_ptr);

} // namespace ElementBinary
} // namespace Kernels
} // namespace FlexFlow

#endif
