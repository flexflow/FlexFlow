#ifndef _FLEXFLOW_OPS_KERNELS_SOFTMAX_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_SOFTMAX_KERNELS_H

#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"
#include "flexflow/ops/softmax.h"

namespace FlexFlow {

class SoftmaxMeta : public OpMeta {
public:
  SoftmaxMeta(FFHandler handle,
              Softmax const *softmax,
              Legion::Domain const &input_domain);
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  cudnnTensorDescriptor_t inputTensor;
  cudnnTensorDescriptor_t outputTensor;
#else
  miopenTensorDescriptor_t inputTensor;
  miopenTensorDescriptor_t outputTensor;
#endif
  bool profiling;
  bool inference_debugging;
  int dim;
  DataType input_type, output_type;
};

namespace Kernels {
namespace Softmax {
template <typename DT>
void forward_kernel_wrapper(SoftmaxMeta const *m,
                            DT const *input_ptr,
                            DT *output_ptr);
template <typename DT>
void backward_kernel_wrapper(SoftmaxMeta const *m,
                             DT *input_grad_ptr,
                             DT const *output_grad_ptr,
                             size_t num_elements);

namespace Internal {
template <typename DT>
void forward_kernel(SoftmaxMeta const *m,
                    DT const *input_ptr,
                    DT *output_ptr,
                    ffStream_t stream);

template <typename DT>
void backward_kernel(DT *input_grad_ptr,
                     DT const *output_grad_ptr,
                     size_t num_elements,
                     ffStream_t stream);
} // namespace Internal
} // namespace Softmax
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_SOFTMAX_KERNELS_H
