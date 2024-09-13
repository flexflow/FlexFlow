#ifndef _FLEXFLOW_OPS_KERNELS_DROPOUT_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_DROPOUT_KERNELS_H

#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"
#include "flexflow/ops/dropout.h"
#include "flexflow/accessor.h"

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
  curandState *state;
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnDropoutDescriptor_t dropoutDesc;
#else
  miopenTensorDescriptor_t inputTensor, outputTensor;
  miopenDropoutDescriptor_t dropoutDesc;
  hiprandState *state;
#endif
  void *reserveSpace, *dropoutStates;
  size_t reserveSpaceSize, dropoutStateSize;
  size_t num_elements;
  long long seed;
  float rate;
};

namespace Kernels {
namespace Dropout {
void forward_kernel_wrapper(DropoutMeta *m,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorW const &output);
void backward_kernel_wrapper(DropoutMeta *m,
                            GenericTensorAccessorR const &output_grad,
                            GenericTensorAccessorW const &input_grad);

namespace Internal {
void forward_kernel(DropoutMeta *m,
                    float const *input_ptr,
                    float *output_ptr,
                    size_t num_elements,
                    ffStream_t stream);
void backward_kernel(DropoutMeta *m,
                     float const *output_grad_ptr,
                     float *input_grad_ptr,
                     size_t num_elements,
                     ffStream_t stream);
} // namespace Internal
} // namespace Dropout
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_DROPOUT_KERNELS_H
