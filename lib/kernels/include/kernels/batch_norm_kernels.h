#ifndef _FLEXFLOW_KERNELS_BATCH_NORM_KERNELS_H
#define _FLEXFLOW_KERNELS_BATCH_NORM_KERNELS_H

#include "device.h"
#include "kernels/allocation.h"
#include "kernels/ff_handle.h"
#include <memory>

namespace FlexFlow {

struct BatchNormPerDeviceState {
  PerDeviceFFHandle handle;
  ffTensorDescriptor_t inputTensor;
  ffTensorDescriptor_t outputTensor;
  ffTensorDescriptor_t biasTensor;
  ffActivationDescriptor_t actiDesc;
  ffBatchNormMode_t mode;
  float *runningMean;
  float *runningVar;
  float *saveMean;
  float *saveVar;
  int output_n;
  int output_c;
  int output_h;
  int output_w;
  req<bool> relu;
};

FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(BatchNormPerDeviceState,
                                             handle,
                                             inputTensor,
                                             outputTensor,
                                             biasTensor,
                                             actiDesc,
                                             mode,
                                             runningMean,
                                             runningVar,
                                             saveMean,
                                             saveVar,
                                             output_n,
                                             output_c,
                                             output_h,
                                             output_w,
                                             relu);

namespace Kernels::BatchNorm {

BatchNormPerDeviceState init_kernel(PerDeviceFFHandle handle,
                                    Allocator allocator,
                                    float *runningMean,
                                    int output_n,
                                    int output_c,
                                    int output_h,
                                    int output_w,
                                    bool relu);

void forward_kernel(ffStream_t stream,
                    BatchNormPerDeviceState const &m,
                    float const *input_ptr,
                    float *output_ptr,
                    float const *scale_ptr,
                    float const *bias_ptr);

void backward_kernel(ffStream_t stream,
                     BatchNormPerDeviceState const &m,
                     float const *input_ptr,
                     float *output_grad_ptr,
                     float const *output_ptr,
                     float *input_grad_ptr,
                     float const *scale_ptr,
                     float *scale_grad_ptr,
                     float *bias_grad_ptr,
                     size_t numElements);

void cleanup_kernel(Allocator allocator,
                    ffTensorDescriptor_t inputTensor,
                    ffTensorDescriptor_t biasTensor,
                    ffTensorDescriptor_t outputTensor,
                    ffActivationDescriptor_t actiDesc,
                    bool relu,
                    float *runningMean);

} // namespace Kernels::BatchNorm
} // namespace FlexFlow

#endif
