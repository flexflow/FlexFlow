#ifndef _FLEXFLOW_OPS_KERNELS_ELEMENT_UNARY_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_ELEMENT_UNARY_KERNELS_H

#include "device.h"
#include "fftype.h"
#include "op_meta.h"

namespace FlexFlow {

class ElementUnaryMeta : public OpMeta {
public:
  ElementUnaryMeta(FFHandler handle);
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnActivationDescriptor_t actiDesc;
#else
  miopenTensorDescriptor_t inputTensor, outputTensor;
  miopenActivationDescriptor_t actiDesc;
#endif
  OperatorType op_type;
  DataType data_type;
  bool inplace;
  float scalar;
  char op_name[MAX_OPNAME];
};

namespace Kernels {
namespace ElementUnary {

void init_kernel(ElementUnaryMeta *m,
                 Legion::Domain const &input_domain,
                 Legion::Domain const &output_domain);

template <typename T>
void forward_kernel_wrapper(ElementUnaryMeta const *m,
                            T const *in_ptr,
                            T *out_ptr,
                            size_t num_elements);
template <typename T>
void backward_kernel_wrapper(ElementUnaryMeta const *m,
                             T const *in_ptr,
                             T *in_grad_ptr,
                             T const *out_ptr,
                             T const *out_grad_ptr,
                             size_t num_elements);

namespace Internal {
template <typename T>
void forward_kernel(ElementUnaryMeta const *m,
                    T const *in_ptr,
                    T *out_ptr,
                    size_t num_elements,
                    ffStream_t stream);
template <typename T>
void backward_kernel(ElementUnaryMeta const *m,
                     T const *in_ptr,
                     T *in_grad_ptr,
                     T const *out_ptr,
                     T const *out_grad_ptr,
                     size_t num_elements,
                     ffStream_t stream);

} // namespace Internal
} // namespace ElementUnary
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_ELEMENT_UNARY_KERNELS_H