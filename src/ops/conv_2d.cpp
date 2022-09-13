/* Copyright 2020 Stanford
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

#include "flexflow/ops/conv_2d.h"
#include "flexflow/utils/hip_helper.h"
#include <hip/hip_runtime.h>

namespace FlexFlow {

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
    void *y);
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
    void *dw);
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
    void *dx);

/*static*/
void Conv2D::init_kernel(Conv2D const *conv,
                         Conv2DMeta *m,
                         int input_w,
                         int input_h,
                         int input_c,
                         int input_n,
                         int output_w,
                         int output_h,
                         int output_c,
                         int output_n,
                         int pad_h,
                         int pad_w,
                         float const *input_ptr,
                         float *output_ptr,
                         float const *kernel_ptr,
                         float *kernel_grad_ptr) {
  checkCUDNN(miopenSet4dTensorDescriptor(
      m->inputTensor, miopenFloat, input_n, input_c, input_h, input_w));

  checkCUDNN(miopenSet4dTensorDescriptor(
      m->biasTensor, miopenFloat, 1, output_c, 1, 1));

  // Require that input_c is divisible by conv->groups
  assert(input_c % conv->groups == 0);
  printf("filterDim: kernel(%d %d) c_in(%d), c_out(%d)\n",
         conv->kernel_h,
         conv->kernel_w,
         input_c / conv->groups,
         output_c);
  checkCUDNN(miopenSet4dTensorDescriptor(m->filterDesc,
                                         miopenFloat,
                                         output_c,
                                         input_c / conv->groups,
                                         conv->kernel_h,
                                         conv->kernel_w));

  checkCUDNN(miopenInitConvolutionDescriptor(m->convDesc,
                                             miopenConvolution,
                                             pad_h, // conv->padding_h,
                                             pad_w, // conv->padding_w,
                                             conv->stride_h,
                                             conv->stride_w,
                                             1 /*upscale_x*/,
                                             1 /*upscale_y*/));
  if (conv->groups != 1) {
    checkCUDNN(miopenSetConvolutionGroupCount(m->convDesc, conv->groups));
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
                                                 output_ptr);
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
                                               kernel_grad_ptr);
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
                                             (float *)input_ptr);
  if (m->relu) {
    checkCUDNN(miopenSetActivationDescriptor(
        m->actiDesc, miopenActivationRELU, 0.0, 0.0, 0.0));
  }
}

/*static*/
void Conv2D::forward_kernel(Conv2DMeta const *m,
                            float const *input_ptr,
                            float *output_ptr,
                            float const *filter_ptr,
                            float const *bias_ptr,
                            hipStream_t stream) {

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

/*static*/
void Conv2D::forward_kernel_wrapper(Conv2DMeta const *m,
                                    float const *input_ptr,
                                    float *output_ptr,
                                    float const *filter_ptr,
                                    float const *bias_ptr) {
  // printf("fwdAlgo(%d), bwdFilterALgo(%d), bwdDataAlgo(%d)\n",
  // (int)m->fwdAlgo,(int) m->bwdFilterAlgo,(int) m->bwdDataAlgo);
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  hipEvent_t t_start, t_end;
  if (m->profiling) {
    hipEventCreate(&t_start);
    hipEventCreate(&t_end);
    hipEventRecord(t_start, stream);
  }

  Conv2D::forward_kernel(
      m, input_ptr, output_ptr, filter_ptr, bias_ptr, stream);
  if (m->profiling) {
    hipEventRecord(t_end, stream);
    checkCUDA(hipEventSynchronize(t_end));
    print_tensor<float>(input_ptr, 16, "[Conv2D:forward:input]");
    print_tensor<float>(filter_ptr, 16, "[Conv2D:forward:kernel]");
    print_tensor<float>(bias_ptr, 16, "[Conv2D:forward:bias]");
    print_tensor<float>(output_ptr, 16, "[Conv2D:forward:output]");
    float elapsed = 0;
    checkCUDA(hipEventElapsedTime(&elapsed, t_start, t_end));
    hipEventDestroy(t_start);
    hipEventDestroy(t_end);
    printf("%s [Conv2D] forward time (CF) = %.2fms\n", m->op_name, elapsed);
  }
}

/*static*/
void Conv2D::backward_kernel(Conv2DMeta const *m,
                             float const *input_ptr,
                             float *input_grad_ptr,
                             float const *output_ptr,
                             float *output_grad_ptr,
                             float const *kernel_ptr,
                             float *kernel_grad_ptr,
                             float *bias_grad_ptr,
                             hipStream_t stream) {

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

/*static*/
void Conv2D::backward_kernel_wrapper(Conv2DMeta const *m,
                                     float const *input_ptr,
                                     float *input_grad_ptr,
                                     float const *output_ptr,
                                     float *output_grad_ptr,
                                     float const *kernel_ptr,
                                     float *kernel_grad_ptr,
                                     float *bias_grad_ptr) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  hipEvent_t t_start, t_end;
  if (m->profiling) {
    hipEventCreate(&t_start);
    hipEventCreate(&t_end);
    hipEventRecord(t_start, stream);
  }

  Conv2D::backward_kernel(m,
                          input_ptr,
                          input_grad_ptr,
                          output_ptr,
                          output_grad_ptr,
                          kernel_ptr,
                          kernel_grad_ptr,
                          bias_grad_ptr,
                          stream);
  if (m->profiling) {
    hipEventRecord(t_end, stream);
    checkCUDA(hipEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(hipEventElapsedTime(&elapsed, t_start, t_end));
    hipEventDestroy(t_start);
    hipEventDestroy(t_end);
    printf("%s [Conv2D] backward time = %.2fms\n", m->op_name, elapsed);
    // print_tensor<4, float>(acc_output_grad.ptr, acc_output_grad.rect,
    // "[Conv2D:backward:output_grad]"); print_tensor<4,
    // float>(acc_kernel_grad.ptr, acc_kernel_grad.rect,
    // "[Conv2D:backward:kernel_grad]"); print_tensor<1,
    // float>(acc_bias_grad.ptr, acc_bias_grad.rect,
    // "[Conv2D:backward:bias_grad]"); print_tensor<4,
    // float>(acc_input_grad.ptr, acc_input_grad.rect,
    // "[Conv2D:backward:input_grad]");
  }
}

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
    void *y) {
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
  printf("forwardAlgo(%d) time(%.2lf)\n",
         perfResults[0].fwd_algo,
         perfResults[0].time);
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
    void *dw) {
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
  printf("bwdFilterAlgo(%d) time(%.2lf)\n",
         perfResults[0].bwd_weights_algo,
         perfResults[0].time);
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
    void *dx) {
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
  printf("bwdDataAlgo(%d) time(%.2lf)\n",
         perfResults[0].bwd_data_algo,
         perfResults[0].time);
  return perfResults[0].bwd_data_algo;
}

Conv2DMeta::Conv2DMeta(FFHandler handler) : OpMeta(handler) {
  checkCUDNN(miopenCreateTensorDescriptor(&inputTensor));
  checkCUDNN(miopenCreateTensorDescriptor(&biasTensor));
  checkCUDNN(miopenCreateTensorDescriptor(&outputTensor));
  checkCUDNN(miopenCreateTensorDescriptor(&filterDesc));
  checkCUDNN(miopenCreateConvolutionDescriptor(&convDesc));
  checkCUDNN(miopenCreateActivationDescriptor(&actiDesc));
}

bool Conv2D::measure_operator_cost(Simulator *sim,
                                   MachineView const &pc,
                                   CostMetrics &cost_metrics) const {
#if 0
  ParallelTensorBase sub_output, sub_input;
  if(!outputs[0]->get_output_sub_tensor(pc, sub_output, OP_CONV2D))
    return false;
  if(!inputs[0]->get_input_sub_tensor(pc, sub_input, OP_CONV2D))
    return false;
  int input_w = sub_input.dims[0].size;
  int input_h = sub_input.dims[1].size;
  int input_c = sub_input.dims[2].size;
  int input_n = sub_input.dims[3].size;
  int output_w = sub_output.dims[0].size;
  int output_h = sub_output.dims[1].size;
  int output_c = sub_output.dims[2].size;
  int output_n = sub_output.dims[3].size;
  int pad_h = ((output_h - 1) * stride_h + kernel_h - input_h + 1) / 2;
  int pad_w = ((output_w - 1) * stride_w + kernel_w - input_w + 1) / 2;

  Conv2DMeta* m = sim->conv2d_meta;
  m->relu = activation == AC_MODE_RELU;
  checkCUDNN(hipdnnSetTensor4dDescriptor(m->inputTensor, HIPDNN_TENSOR_NCHW,
      HIPDNN_DATA_FLOAT, input_n, input_c, input_h, input_w));
  checkCUDNN(hipdnnSetTensor4dDescriptor(m->biasTensor, HIPDNN_TENSOR_NCHW,
      HIPDNN_DATA_FLOAT, 1, output_c, 1, 1));
  // require input_c is divisible by groups
  assert(input_c % groups == 0);
  checkCUDNN(hipdnnSetFilter4dDescriptor(m->filterDesc, HIPDNN_DATA_FLOAT,
      HIPDNN_TENSOR_NCHW, output_c, input_c / groups, kernel_h, kernel_w));
  checkCUDNN(hipdnnSetConvolution2dDescriptor(m->convDesc, pad_h, pad_w,
      stride_h, stride_w, 1/*dilationH*/, 1/*dilationW*/,
      HIPDNN_CROSS_CORRELATION, HIPDNN_DATA_FLOAT));

  checkCUDNN(hipdnnSetConvolutionGroupCount(m->convDesc, groups));
  if (m->handle.allowTensorOpMathConversion) {
    checkCUDNN(hipdnnSetConvolutionMathType(m->convDesc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
  } else {
    checkCUDNN(hipdnnSetConvolutionMathType(m->convDesc, HIPDNN_TENSOR_OP_MATH));
  }
  int n, c, h, w;
  checkCUDNN(hipdnnGetConvolution2dForwardOutputDim(m->convDesc,
      m->inputTensor, m->filterDesc, &n, &c, &h, &w));
  assert(n == output_n);
  assert(c == output_c);
  assert(h == output_h);
  assert(w == output_w);
  checkCUDNN(hipdnnSetActivationDescriptor(m->actiDesc, HIPDNN_ACTIVATION_RELU,
      HIPDNN_NOT_PROPAGATE_NAN, 0.0));
  checkCUDNN(hipdnnSetTensor4dDescriptor(m->outputTensor, HIPDNN_TENSOR_NCHW,
      HIPDNN_DATA_FLOAT, n, c, h, w));
  // allocate tensors in simulator
  sim->free_all();
  float* input_ptr = (float*)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  assert(input_ptr != NULL);
  float *output_ptr = (float*)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert(output_ptr != NULL);
  float* weight_ptr = (float*)sim->allocate((size_t)output_c * input_c * kernel_h * kernel_w / groups, DT_FLOAT);
  assert(weight_ptr != NULL);
  float* bias_ptr = (float*)sim->allocate(output_c, DT_FLOAT);
  assert(bias_ptr != NULL);

  // select forward algorithm
  {
    const int reqAlgCnt = 8;
    int cnt = 0;
    hipdnnConvolutionFwdAlgoPerf_t perfResults[reqAlgCnt];
    checkCUDNN(hipdnnFindConvolutionForwardAlgorithmEx(
        m->handle.dnn, m->inputTensor, input_ptr,
        m->filterDesc, weight_ptr, m->convDesc, m->outputTensor, output_ptr,
        reqAlgCnt, &cnt, perfResults,
        m->handle.workSpace, m->handle.workSpaceSize));
    assert(cnt > 0);
    checkCUDNN(perfResults[0].status);
    cost_metrics.forward_time = perfResults[0].time;
    //for (int i = 0; i < cnt; i++)
    //  printf("conv forward: algo(%d) time(%.4lf)\n", perfResults[i].algo, perfResults[i].time);
  }
  // select backward algorithm
  {
    const int reqAlgCnt = 8;
    int cnt = 0;
    hipdnnConvolutionBwdFilterAlgoPerf_t perfResults[reqAlgCnt];
    checkCUDNN(hipdnnFindConvolutionBackwardFilterAlgorithmEx(
        m->handle.dnn, m->inputTensor, input_ptr,
        m->outputTensor, output_ptr, m->convDesc, m->filterDesc, weight_ptr,
        reqAlgCnt, &cnt, perfResults,
        m->handle.workSpace, m->handle.workSpaceSize));
    assert(cnt > 0);
    checkCUDNN(perfResults[0].status);
    cost_metrics.backward_time = perfResults[0].time;
    //for (int i = 0; i < cnt; i++)
    //  printf("conv backward filter: algo(%d) time(%.4lf)\n", perfResults[i].algo, perfResults[i].time);
  }
  if (trainableInputs[0]) {
    const int reqAlgCnt = 8;
    int cnt = 0;
    hipdnnConvolutionBwdDataAlgoPerf_t perfResults[reqAlgCnt];
    checkCUDNN(hipdnnFindConvolutionBackwardDataAlgorithmEx(
        m->handle.dnn, m->filterDesc, weight_ptr,
        m->outputTensor, output_ptr, m->convDesc, m->inputTensor, input_ptr,
        reqAlgCnt, &cnt, perfResults,
        m->handle.workSpace, m->handle.workSpaceSize));
    assert(cnt > 0);
    checkCUDNN(perfResults[0].status);
    cost_metrics.backward_time += perfResults[0].time;
    //for (int i = 0; i < cnt; i++)
    //  printf("conv backward data: algo(%d) time(%.4lf)\n", perfResults[i].algo, perfResults[i].time);
  }
  log_measure.debug("[Measure Conv2D] name(%s) input(%d %d %d %d) weight(%d %d %d %d) output(%d %d %d %d) stride(%d %d) padding(%d %d) forward_time(%.4lf) backward_time(%.4lf)\n",
         name,
         input_n, input_c, input_h, input_w,
         output_c, input_c / groups, kernel_h, kernel_w,
         output_n, output_c, output_h, output_w,
         stride_h, stride_w,
         padding_h, padding_w,
         cost_metrics.forward_time, cost_metrics.backward_time);
#endif
  return true;
}

}; // namespace FlexFlow
