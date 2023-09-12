/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernels/conv_2d_kernels.h"

namespace FlexFlow {
namespace Kernels {
namespace Conv2D {

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
    float *time) {
  int const reqAlgCnt = 8;
  int cnt = 0;
  miopenConvAlgoPerf_t perfResults[reqAlgCnt];
  checkCUDNN(miopenFindConvolutionForwardAlgorithm(handle,
                                                   xDesc,
                                                   x,
                                                   wDesc,
                                                   w,
                                                   convDesc,
                                                   yDesc,
                                                   y,
                                                   reqAlgCnt,
                                                   &cnt,
                                                   perfResults,
                                                   workSpace,
                                                   workSpaceSize,
                                                   false));
  assert(cnt > 0);
  // checkCUDNN(perfResults[0].status);
  if (time != nullptr) {
    *time = perfResults[0].time;
  }
  return perfResults[0].fwd_algo;
}

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
    float *time) {
  int const reqAlgCnt = 8;
  int cnt = 0;
  miopenConvAlgoPerf_t perfResults[reqAlgCnt];
  checkCUDNN(miopenFindConvolutionBackwardWeightsAlgorithm(handle,
                                                           dyDesc,
                                                           dy,
                                                           xDesc,
                                                           x,
                                                           convDesc,
                                                           dwDesc,
                                                           dw,
                                                           reqAlgCnt,
                                                           &cnt,
                                                           perfResults,
                                                           workSpace,
                                                           workSpaceSize,
                                                           false));
  assert(cnt > 0);
  // checkCUDNN(perfResults[0].status);
  if (time != nullptr) {
    *time = perfResults[0].time;
  }
  return perfResults[0].bwd_weights_algo;
}

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
    float *time) {
  int const reqAlgCnt = 8;
  int cnt = 0;
  miopenConvAlgoPerf_t perfResults[reqAlgCnt];
  checkCUDNN(miopenFindConvolutionBackwardDataAlgorithm(handle,
                                                        dyDesc,
                                                        dy,
                                                        wDesc,
                                                        w,
                                                        convDesc,
                                                        dxDesc,
                                                        dx,
                                                        reqAlgCnt,
                                                        &cnt,
                                                        perfResults,
                                                        workSpace,
                                                        workSpaceSize,
                                                        false));
  assert(cnt > 0);
  // checkCUDNN(perfResults[0].status);
  if (time != nullptr) {
    *time = perfResults[0].time;
  }
  return perfResults[0].bwd_data_algo;
}

Conv2DPerDeviceState init_kernel(PerDeviceFFHandle handle,
                                 optional<Activation> activation,
                                 int kernel_h,
                                 int kernel_w,
                                 int groups,
                                 int padding_h,
                                 int padding_w,
                                 int stride_h,
                                 int stride_w,
                                 GenericTensorAccessorR const &input,
                                 GenericTensorAccessorW const &output,
                                 float const *filter_ptr,
                                 float *filter_grad_ptr) {

  miopenCreateTensorDescriptor inputTensor;
  miopenCreateTensorDescriptor biasTensor;
  miopenCreateTensorDescriptor outputTensor;
  miopenCreateTensorDescriptor filterDesc;
  miopenActivationDescriptor_t actiDesc;
  miopenActivationDescriptor_t convDesc;
  miopenConvFwdAlgorithm_t fwdAlgo;
  miopenConvBwdWeightsAlgorithm_t bwdFilterAlgo;
  miopenConvBwdDataAlgorithm_t bwdDataAlgo;

  int input_w = input.shape[legion_dim_t(0)];
  int input_h = input.shape[legion_dim_t(1)];
  int input_c = input.shape[legion_dim_t(2)];
  int input_n = input.shape[legion_dim_t(3)];

  int output_w = output.shape[legion_dim_t(0)];
  int output_h = output.shape[legion_dim_t(1)];
  int output_c = output.shape[legion_dim_t(2)];
  int output_n = output.shape[legion_dim_t(3)];

  checkCUDNN(miopenCreateTensorDescriptor(&inputTensor));
  checkCUDNN(miopenCreateTensorDescriptor(&biasTensor));
  checkCUDNN(miopenCreateTensorDescriptor(&outputTensor));
  checkCUDNN(miopenCreateTensorDescriptor(&filterDesc));
  checkCUDNN(miopenCreateConvolutionDescriptor(&convDesc));
  checkCUDNN(miopenCreateActivationDescriptor(&actiDesc));

  checkCUDNN(miopenSet4dTensorDescriptor(
      inputTensor, miopenFloat, input_n, input_c, input_h, input_w));

  checkCUDNN(
      miopenSet4dTensorDescriptor(biasTensor, miopenFloat, 1, output_c, 1, 1));

  // Require that input_c is divisible by conv->groups
  assert(input_c % groups == 0);
  checkCUDNN(miopenSet4dTensorDescriptor(m->filterDesc,
                                         miopenFloat,
                                         output_c,
                                         input_c / groups,
                                         kernel_h,
                                         kernel_w));

  checkCUDNN(miopenInitConvolutionDescriptor(m->convDesc,
                                             miopenConvolution,
                                             pad_h, // conv->padding_h,
                                             pad_w, // conv->padding_w,
                                             stride_h,
                                             stride_w,
                                             1 /*upscale_x*/,
                                             1 /*upscale_y*/));
  if (groups != 1) {
    checkCUDNN(miopenSetConvolutionGroupCount(m->convDesc, groups));
  }

  // TODO: enable tensor core when possible
  if (m->handle.allowTensorOpMathConversion) {
    // checkCUDNN(hipdnnSetConvolutionMathType(m->convDesc,
    // CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
  } else {
    // checkCUDNN(hipdnnSetConvolutionMathType(m->convDesc,
    // HIPDNN_TENSOR_OP_MATH));
  }

  int n, c, h, w;
  checkCUDNN(miopenGetConvolutionForwardOutputDim(
      m->convDesc, m->inputTensor, m->filterDesc, &n, &c, &h, &w));
  assert(n == output_n);
  assert(c == output_c);
  assert(h == output_h);
  assert(w == output_w);

  checkCUDNN(
      miopenSet4dTensorDescriptor(m->outputTensor, miopenFloat, n, c, h, w));

  float time;
  // select forward algorithm
  m->fwdAlgo = selectConvolutionForwardAlgorithm(m->handle.dnn,
                                                 m->inputTensor,
                                                 input_ptr,
                                                 m->filterDesc,
                                                 kernel_ptr,
                                                 m->convDesc,
                                                 m->handle.workSpace,
                                                 m->handle.workSpaceSize,
                                                 m->outputTensor,
                                                 output_ptr,
                                                 &time);
  if (forward_time != nullptr) {
    *forward_time += time;
  }

  // select backward filter algorithm
  m->bwdFilterAlgo =
      selectConvolutionBackwardFilterAlgorithm(m->handle.dnn,
                                               m->inputTensor,
                                               input_ptr,
                                               m->outputTensor,
                                               output_ptr,
                                               m->convDesc,
                                               m->handle.workSpace,
                                               m->handle.workSpaceSize,
                                               m->filterDesc,
                                               kernel_grad_ptr,
                                               &time);
  if (backward_time != nullptr) {
    *backward_time += time;
  }

  // select backward data algorithm
  m->bwdDataAlgo =
      selectConvolutionBackwardDataAlgorithm(m->handle.dnn,
                                             m->filterDesc,
                                             kernel_ptr,
                                             m->outputTensor,
                                             output_ptr,
                                             m->convDesc,
                                             m->handle.workSpace,
                                             m->handle.workSpaceSize,
                                             m->inputTensor,
                                             (float *)input_ptr,
                                             &time);
  if (backward_time != nullptr) {
    *backward_time += time;
  }
  if (m->relu) {
    checkCUDNN(miopenSetActivationDescriptor(
        m->actiDesc, miopenActivationRELU, 0.0, 0.0, 0.0));
  }
}

void forward_kernel(ffStream_t stream,
                    Conv2DPerDeviceState const &m,
                    float const *input_ptr,
                    float *output_ptr,
                    float const *filter_ptr,
                    float const *bias_ptr,
                    optional<Activation> activation) {

  checkCUDNN(miopenSetStream(m->handle.dnn, stream));

  float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(miopenConvolutionForward(m->handle.dnn,
                                      &alpha,
                                      m->inputTensor,
                                      input_ptr,
                                      m->filterDesc,
                                      filter_ptr,
                                      m->convDesc,
                                      m->fwdAlgo,
                                      &beta,
                                      m->outputTensor,
                                      output_ptr,
                                      m->handle.workSpace,
                                      m->handle.workSpaceSize));

  // use_bias == True
  if (bias_ptr != NULL) {
    checkCUDNN(miopenConvolutionForwardBias(m->handle.dnn,
                                            &alpha,
                                            m->biasTensor,
                                            bias_ptr,
                                            &alpha,
                                            m->outputTensor,
                                            output_ptr));
  }
  if (m->relu) {
    checkCUDNN(miopenActivationForward(m->handle.dnn,
                                       m->actiDesc,
                                       &alpha,
                                       m->outputTensor,
                                       output_ptr,
                                       &beta,
                                       m->outputTensor,
                                       output_ptr));
  }
}

void backward_kernel(ffStream_t stream,
                     Conv2DPerDeviceState const &m,
                     float const *input_ptr,
                     float *input_grad_ptr,
                     float const *output_ptr,
                     float *output_grad_ptr,
                     float const *filter_ptr,
                     float *filter_grad_ptr,
                     float *bias_grad_ptr,
                     optional<Activation> activation) {

  checkCUDNN(miopenSetStream(m->handle.dnn, stream));

  float alpha = 1.0f;
  float beta = 0.0f;
  if (m->relu) {
    miopenDataType_t dataType;
    int n, c, h, w, nStride, cStride, hStride, wStride;
    checkCUDNN(miopenGet4dTensorDescriptor(m->outputTensor,
                                           &dataType,
                                           &n,
                                           &c,
                                           &h,
                                           &w,
                                           &nStride,
                                           &cStride,
                                           &hStride,
                                           &wStride));
    hipLaunchKernelGGL(reluBackward,
                       GET_BLOCKS(n * c * h * w),
                       CUDA_NUM_THREADS,
                       0,
                       stream,
                       output_grad_ptr,
                       output_ptr,
                       n * c * h * w);
  }
  // Compute filter gradiant
  // NOTE: we use alpha for kernel_grad to accumulate gradients
  checkCUDNN(miopenConvolutionBackwardWeights(m->handle.dnn,
                                              &alpha,
                                              m->outputTensor,
                                              output_grad_ptr,
                                              m->inputTensor,
                                              input_ptr,
                                              m->convDesc,
                                              m->bwdFilterAlgo,
                                              &beta,
                                              m->filterDesc,
                                              kernel_grad_ptr,
                                              m->handle.workSpace,
                                              m->handle.workSpaceSize));
  // Compute bias gradiant
  // NOTE: we use alpha for bias_grad to accumulate gradients
  if (bias_grad_ptr != NULL) {
    checkCUDNN(miopenConvolutionBackwardBias(m->handle.dnn,
                                             &alpha,
                                             m->outputTensor,
                                             output_grad_ptr,
                                             &beta,
                                             m->biasTensor,
                                             bias_grad_ptr));
  }
  // Compute data gradiant
  // NOTE: we use alpha for input_grad to accumulate gradients
  if (input_grad_ptr != NULL) {
    checkCUDNN(miopenConvolutionBackwardData(m->handle.dnn,
                                             &alpha,
                                             m->outputTensor,
                                             output_grad_ptr,
                                             m->filterDesc,
                                             kernel_ptr,
                                             m->convDesc,
                                             m->bwdDataAlgo,
                                             &beta,
                                             m->inputTensor,
                                             input_grad_ptr,
                                             m->handle.workSpace,
                                             m->handle.workSpaceSize));
  }
}

} // namespace Conv2D
} // namespace Kernels
} // namespace FlexFlow
