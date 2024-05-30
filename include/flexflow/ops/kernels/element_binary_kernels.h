#ifndef _FLEXFLOW_OPS_KERNELS_ELEMENT_BINARY_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_ELEMENT_BINARY_KERNELS_H

#include "flexflow/accessor.h"
#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"

namespace FlexFlow {

class ElementBinaryMeta : public OpMeta {
public:
  ElementBinaryMeta(FFHandler handle, Op const *op);
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  cudnnTensorDescriptor_t input1Tensor, input2Tensor, outputTensor;
  cudnnOpTensorDescriptor_t opDesc;
  cudnnReduceTensorDescriptor_t reduceAddDesc;
#else
  miopenTensorDescriptor_t input1Tensor, input2Tensor, outputTensor;
  miopenTensorOp_t opDesc;
  miopenReduceTensorDescriptor_t reduceAddDesc;
#endif
  OperatorType op_type;
  bool inplace_a, has_same_operands;
  bool broadcast_input1, broadcast_input2;
};

namespace Kernels {
namespace ElementBinary {

void init_kernel(ElementBinaryMeta *m,
                 Legion::Domain const &input1_domain,
                 Legion::Domain const &input2_domain,
                 Legion::Domain const &output_domain);

void forward_kernel_wrapper(ElementBinaryMeta const *m,
                            GenericTensorAccessorR const &in1,
                            GenericTensorAccessorR const &in2,
                            GenericTensorAccessorW const &out);

void backward_kernel_wrapper(ElementBinaryMeta const *m,
                             float const *out_grad_ptr,
                             float const *in1_ptr,
                             float const *in2_ptr,
                             float *in1_grad_ptr,
                             float *in2_grad_ptr);

namespace Internal {

template <typename DT>
void forward_kernel(ElementBinaryMeta const *m,
                    DT const *in1_ptr,
                    DT const *in2_ptr,
                    DT *out_ptr,
                    ffStream_t stream);
void backward_kernel(ElementBinaryMeta const *m,
                     float const *out_grad_ptr,
                     float const *in1_ptr,
                     float const *in2_ptr,
                     float *in1_grad_ptr,
                     float *in2_grad_ptr,
                     ffStream_t stream);

} // namespace Internal
} // namespace ElementBinary
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_ELEMENT_BINARY_KERNELS_H
