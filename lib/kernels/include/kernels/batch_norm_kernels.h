#ifndef _FLEXFLOW_KERNELS_BATCH_NORM_KERNELS_H
#define _FLEXFLOW_KERNELS_BATCH_NORM_KERNELS_H

#include "kernels/allocation.h"
#include "kernels/device.h"
#include "kernels/ff_handle.h"
#include <memory>

namespace FlexFlow {

class BatchNormPerDeviceState : public PerDeviceOpState {
public:
  BatchNormPerDeviceState(FFHandler handle,
                          std::unique_ptr<IAllocator> allocator,
                          int output_n,
                          int output_c,
                          int output_h,
                          int output_w,
                          bool relu,
                          bool profiling);
  ~BatchNormPerDeviceState(void);

  ffTensorDescriptor_t inputTensor, outputTensor, biasTensor;
  ffActivationDescriptor_t actiDesc;
  ffBatchNormMode_t mode;
  float *runningMean, *runningVar, *saveMean, *saveVar;
  bool relu;
  bool profiling;
  std::unique_ptr<IAllocator> allocator;
};

namespace Kernels {
namespace BatchNorm {

void forward_kernel(ffStream_t stream,
                    BatchNormPerDeviceState *m,
                    float const *input_ptr,
                    float *output_ptr,
                    float const *scale_ptr,
                    float const *bias_ptr);

void backward_kernel(ffStream_t stream,
                     BatchNormPerDeviceState *m,
                     float const *input_ptr,
                     float *output_grad_ptr,
                     float const *output_ptr,
                     float *input_grad_ptr,
                     float const *scale_ptr,
                     float *scale_grad_ptr,
                     float *bias_grad_ptr,
                     size_t numElements);

} // namespace BatchNorm
} // namespace Kernels
} // namespace FlexFlow

#endif
