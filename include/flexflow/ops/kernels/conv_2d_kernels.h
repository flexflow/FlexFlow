#ifndef _FLEXFLOW_OPS_KERNELS_CONV_2D_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_CONV_2D_KERNELS_H

#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"

namespace FlexFlow {

class Conv2DMeta : public OpMeta {
public:
  Conv2DMeta(FFHandler handler);
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  cudnnTensorDescriptor_t inputTensor, biasTensor, outputTensor;
  cudnnFilterDescriptor_t filterDesc;
  cudnnActivationDescriptor_t actiDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnConvolutionFwdAlgo_t fwdAlgo;
  cudnnConvolutionBwdFilterAlgo_t bwdFilterAlgo;
  cudnnConvolutionBwdDataAlgo_t bwdDataAlgo;
#else
  miopenTensorDescriptor_t inputTensor, biasTensor, outputTensor;
  miopenTensorDescriptor_t filterDesc;
  miopenActivationDescriptor_t actiDesc;
  miopenConvolutionDescriptor_t convDesc;
  miopenConvFwdAlgorithm_t fwdAlgo;
  miopenConvBwdWeightsAlgorithm_t bwdFilterAlgo;
  miopenConvBwdDataAlgorithm_t bwdDataAlgo;
#endif
  bool relu, use_bias;
};

namespace Kernels {
namespace Conv2D {

void init_kernel(Conv2DMeta *m,
                 int input_w,
                 int input_h,
                 int input_c,
                 int input_n,
                 int output_w,
                 int output_h,
                 int output_c,
                 int output_n,
                 int kernel_h,
                 int kernel_w,
                 int groups,
                 int stride_h,
                 int stride_w,
                 int pad_h,
                 int pad_w,
                 float const *input_ptr,
                 float *output_ptr,
                 float const *kernel_ptr,
                 float *kernel_grad_ptr,
                 float *forward_time = nullptr,
                 float *backward_time = nullptr);

void forward_kernel_wrapper(Conv2DMeta const *m,
                            float const *input_ptr,
                            float *output_ptr,
                            float const *filter_ptr,
                            float const *bias_ptr);
void backward_kernel_wrapper(Conv2DMeta const *m,
                             float const *input_ptr,
                             float *input_grad_ptr,
                             float const *output_ptr,
                             float *output_grad_ptr,
                             float const *kernel_ptr,
                             float *kernel_grad_ptr,
                             float *bias_grad_ptr);

namespace Internal {

void forward_kernel(Conv2DMeta const *m,
                    float const *input_ptr,
                    float *output_ptr,
                    float const *filter_ptr,
                    float const *bias_ptr,
                    ffStream_t stream);

void backward_kernel(Conv2DMeta const *m,
                     float const *input_ptr,
                     float *input_grad_ptr,
                     float const *output_ptr,
                     float *output_grad_ptr,
                     float const *kernel_ptr,
                     float *kernel_grad_ptr,
                     float *bias_grad_ptr,
                     ffStream_t stream);

#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
cudnnConvolutionFwdAlgo_t selectConvolutionForwardAlgorithm(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    void const *x,
    const cudnnFilterDescriptor_t wDesc,
    void const *w,
    const cudnnConvolutionDescriptor_t convDesc,
    void *workSpace,
    size_t workSpaceSize,
    const cudnnTensorDescriptor_t yDesc,
    void *y,
    float *time);

cudnnConvolutionBwdDataAlgo_t selectConvolutionBackwardDataAlgorithm(
    cudnnHandle_t handle,
    const cudnnFilterDescriptor_t wDesc,
    void const *w,
    const cudnnTensorDescriptor_t dyDesc,
    void const *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    void *workSpace,
    size_t workSpaceSize,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx,
    float *time);

cudnnConvolutionBwdFilterAlgo_t selectConvolutionBackwardFilterAlgorithm(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    void const *x,
    const cudnnTensorDescriptor_t dyDesc,
    void const *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    void *workSpace,
    size_t workSpaceSize,
    const cudnnFilterDescriptor_t dwDesc,
    void *dw,
    float *time);
#else
miopenConvFwdAlgorithm_t selectConvolutionForwardAlgorithm(
    miopenHandle_t handle,
    const miopenTensorDescriptor_t xDesc,
    void const *x,
    const miopenTensorDescriptor_t wDesc,
    void const *w,
    const miopenConvolutionDescriptor_t convDesc,
    void *workSpace,
    size_t workSpaceSize,
    const miopenTensorDescriptor_t yDesc,
    void *y,
    float *time);

miopenConvBwdWeightsAlgorithm_t selectConvolutionBackwardFilterAlgorithm(
    miopenHandle_t handle,
    const miopenTensorDescriptor_t xDesc,
    void const *x,
    const miopenTensorDescriptor_t dyDesc,
    void const *dy,
    const miopenConvolutionDescriptor_t convDesc,
    void *workSpace,
    size_t workSpaceSize,
    const miopenTensorDescriptor_t dwDesc,
    void *dw,
    float *time);

miopenConvBwdDataAlgorithm_t selectConvolutionBackwardDataAlgorithm(
    miopenHandle_t handle,
    const miopenTensorDescriptor_t wDesc,
    void const *w,
    const miopenTensorDescriptor_t dyDesc,
    void const *dy,
    const miopenConvolutionDescriptor_t convDesc,
    void *workSpace,
    size_t workSpaceSize,
    const miopenTensorDescriptor_t dxDesc,
    void *dx,
    float *time);
#endif

} // namespace Internal
} // namespace Conv2D
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_CONV_2D_KERNELS_H
