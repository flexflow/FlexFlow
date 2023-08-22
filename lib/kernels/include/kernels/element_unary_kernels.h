#ifndef _FLEXFLOW_OPS_KERNELS_ELEMENT_UNARY_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_ELEMENT_UNARY_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device.h"
#include "legion.h"
#include <cstddef>

namespace FlexFlow {

class ElementUnaryPerDeviceState : public PerDeviceOpState {
public:
  ElementUnaryPerDeviceState(FFHandler handle);
  ffTensorDescriptor_t inputTensor, outputTensor;
  ffActivationDescriptor_t actiDesc;

  OperatorType op_type;
  DataType data_type;
  bool inplace;
  float scalar;
  char op_name[MAX_OPNAME];
};

namespace Kernels {
namespace ElementUnary {

void init_kernel(ElementUnaryPerDeviceState *m,
                 Legion::Domain const &input_domain,
                 Legion::Domain const &output_domain);

void forward_kernel(ffStream_t stream,
                    ElementUnaryPerDeviceState const *m,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output);

void backward_kernel(ffStream_t stream,
                     ElementUnaryPerDeviceState const *m,
                     GenericTensorAccessorR const &input,
                     GenericTensorAccessorR const &input_grad,
                     GenericTensorAccessorW const &output,
                     GenericTensorAccessorW const &output_grad);

} // namespace ElementUnary
} // namespace Kernels
} // namespace FlexFlow

#endif
