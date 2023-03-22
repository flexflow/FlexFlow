#ifndef _FLEXFLOW_OPS_KERNELS_DROPOUT_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_DROPOUT_KERNELS_H

#include "kernels/device.h"
#include "kernels/op_meta.h"
#include "legion.h"
#include <cstddef>

namespace FlexFlow {

class DropoutMeta : public OpMeta {
public:
  DropoutMeta(FFHandler handler,
              float rate,
              unsigned long long seed,
              bool profiling,
              Legion::Memory gpu_mem,
              Legion::Domain const &output_domain);
  ~DropoutMeta(void);
  Realm::RegionInstance reserveInst;
  ffTensorDescriptor_t inputTensor, outputTensor;
  ffDropoutDescriptor_t dropoutDesc;
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
