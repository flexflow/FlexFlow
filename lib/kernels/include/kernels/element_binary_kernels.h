#ifndef _FLEXFLOW_OPS_KERNELS_ELEMENT_BINARY_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_ELEMENT_BINARY_KERNELS_H

#include "kernels/device.h"
#include "kernels/op_meta.h"
#include "kernels/domain.h"

namespace FlexFlow {

class ElementBinaryMeta : public OpMeta {
public:
  ElementBinaryMeta(FFHandler handle);
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

void init_kernel(ElementBinaryMeta *m,
                 Domain const &input1_domain,
                 Domain const &input2_domain,
                 Domain const &output_domain);

void forward_kernel_wrapper(ElementBinaryMeta const *m,
                            float const *in1_ptr,
                            float const *in2_ptr,
                            float *out_ptr);

void backward_kernel_wrapper(ElementBinaryMeta const *m,
                             float const *out_grad_ptr,
                             float const *in1_ptr,
                             float const *in2_ptr,
                             float *in1_grad_ptr,
                             float *in2_grad_ptr);

namespace Internal {

void forward_kernel(ElementBinaryMeta const *m,
                    float const *in1_ptr,
                    float const *in2_ptr,
                    float *out_ptr,
                    ffStream_t stream);
void backward_kernel(ElementBinaryMeta const *m,
                     float const *out_grad_ptr,
                     float const *in1_ptr,
                     float const *in2_ptr,
                     float *in1_grad_ptr,
                     float *in2_grad_ptr,
                     ffStream_t stream);

} 
}
}
}

#endif
