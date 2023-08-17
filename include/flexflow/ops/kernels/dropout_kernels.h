#ifndef _FLEXFLOW_OPS_KERNELS_DROPOUT_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_DROPOUT_KERNELS_H

#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"
#include "flexflow/ops/dropout.h"

namespace FlexFlow {

class DropoutMeta : public OpMeta {
public:
  DropoutMeta(FFHandler handle,
              Dropout const *dropout,
              Legion::Memory gpu_mem,
              Legion::Domain const &output_domain);
  ~DropoutMeta(void);
  Realm::RegionInstance reserveInst;
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnDropoutDescriptor_t dropoutDesc;
#else
  miopenTensorDescriptor_t inputTensor, outputTensor;
  miopenDropoutDescriptor_t dropoutDesc;
#endif
  void *reserveSpace, *dropoutStates;
  size_t reserveSpaceSize, dropoutStateSize;
};

namespace Kernels {
namespace Dropout {
void forward_kernel_wrapper(DropoutMeta *m,
                            float const *input_ptr,
                            float *output_ptr);
void backward_kernel_wrapper(DropoutMeta *m,
                             float const *output_grad_ptr,
                             float *input_grad_ptr);

namespace Internal {
void forward_kernel(DropoutMeta *m,
                    float const *input_ptr,
                    float *output_ptr,
                    ffStream_t stream);
void backward_kernel(DropoutMeta *m,
                     float const *output_grad_ptr,
                     float *input_grad_ptr,
                     ffStream_t stream);
} // namespace Internal
} // namespace Dropout
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_DROPOUT_KERNELS_H
