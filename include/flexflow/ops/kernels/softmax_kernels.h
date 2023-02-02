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
#else
  miopenTensorDescriptor_t inputTensor;
#endif
  bool profiling;
  int dim;
  char op_name[MAX_OPNAME];
};

namespace Kernels {
namespace Softmax {

void forward_kernel_wrapper(SoftmaxMeta const *m,
                            float const *input_ptr,
                            float *output_ptr);

void backward_kernel_wrapper(SoftmaxMeta const *m,
                             float *input_grad_ptr,
                             float const *output_grad_ptr,
                             size_t num_elements);

namespace Internal {
void forward_kernel(SoftmaxMeta const *m,
                    float const *input_ptr,
                    float *output_ptr,
                    ffStream_t stream);
void backward_kernel(float *input_grad_ptr,
                     float const *output_grad_ptr,
                     size_t num_elements,
                     ffStream_t stream);
} // namespace Internal
} // namespace Softmax
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_SOFTMAX_KERNELS_H
