#ifndef _FLEXFLOW_OPS_KERNELS_CONV_2D_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_CONV_2D_KERNELS_H

#include "kernels/device.h"

namespace FlexFlow {

struct Conv2DPerDeviceState {
  PerDeviceFFHandle handle;
  ffTensorDescriptor_t inputTensor;
  ffTensorDescriptor_t biasTensor;
  ffTensorDescriptor_t outputTensor;
  ffFilterDescriptor_t filterDesc;
  ffActivationDescriptor_t actiDesc;
  ffConvolutionDescriptor_t convDesc;
  ffConvolutionFwdAlgo_t fwdAlgo;
  ffConvolutionBwdFilterAlgo_t bwdFilterAlgo;
  ffConvolutionBwdDataAlgo_t bwdDataAlgo;
  req<optional<Activation>> activation;
  req<bool> use_bias;
};

FF_VISITABLE_STRUCT_NO_EQ(Conv2DPerDeviceState,
                          handle,
                          inputTensor,
                          biasTensor,
                          outputTensor,
                          filterDesc,
                          actiDesc,
                          convDesc,
                          fwdAlgo,
                          bwdFilterAlgo,
                          bwdDataAlgo,
                          activation,
                          use_bias);

namespace Kernels {
namespace Conv2D {

Conv2DPerDeviceState init_kernel(PerDeviceFFHandle handle,
                                 ffTensorDescriptor_t inputTensor,
                                 ffTensorDescriptor_t biasTensor,
                                 ffTensorDescriptor_t outputTensor,
                                 ffFilterDescriptor_t filterDesc,
                                 ffActivationDescriptor_t actiDesc,
                                 ffConvolutionDescriptor_t convDesc,
                                 ffConvolutionFwdAlgo_t fwdAlgo,
                                 ffConvolutionBwdFilterAlgo_t bwdFilterAlgo,
                                 ffConvolutionBwdDataAlgo_t bwdDataAlgo,
                                 req<optional<Activation>> relu,
                                 bool use_bias);

void forward_kernel(ffStream_t stream,
                    Conv2DPerDeviceState const *m,
                    float const *input_ptr,
                    float *output_ptr,
                    float const *filter_ptr,
                    float const *bias_ptr);

void backward_kernel(ffStream_t stream,
                     Conv2DPerDeviceState const *m,
                     float const *input_ptr,
                     float *input_grad_ptr,
                     float const *output_ptr,
                     float *output_grad_ptr,
                     float const *filter_ptr,
                     float *filter_grad_ptr,
                     float *bias_grad_ptr);

} // namespace Conv2D
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_CONV_2D_KERNELS_H
