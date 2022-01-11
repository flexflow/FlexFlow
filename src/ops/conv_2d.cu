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
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

cudnnConvolutionFwdAlgo_t
selectConvolutionForwardAlgorithm(cudnnHandle_t handle,
                                  const cudnnTensorDescriptor_t xDesc, const void* x,
                                  const cudnnFilterDescriptor_t wDesc, const void* w,
                                  const cudnnConvolutionDescriptor_t convDesc,
                                  void* workSpace, size_t workSpaceSize,
                                  const cudnnTensorDescriptor_t yDesc, void* y);
cudnnConvolutionBwdFilterAlgo_t
selectConvolutionBackwardFilterAlgorithm(cudnnHandle_t handle,
                                         const cudnnTensorDescriptor_t xDesc, const void* x,
                                         const cudnnTensorDescriptor_t dyDesc, const void* dy,
                                         const cudnnConvolutionDescriptor_t convDesc,
                                         void* workSpace, size_t workSpaceSize,
                                         const cudnnFilterDescriptor_t dwDesc, void* dw);
cudnnConvolutionBwdDataAlgo_t
selectConvolutionBackwardDataAlgorithm(cudnnHandle_t handle,
                                       const cudnnFilterDescriptor_t wDesc, const void* w,
                                       const cudnnTensorDescriptor_t dyDesc, const void* dy,
                                       const cudnnConvolutionDescriptor_t convDesc,
                                       void* workSpace, size_t workSpaceSize,
                                       const cudnnTensorDescriptor_t dxDesc, void* dx);

/*static*/
void Conv2D::init_task_kernel(const Conv2D *conv, 
                              Conv2DMeta *m,
                              int input_w, int input_h, int input_c, int input_n,
                              int output_w, int output_h, int output_c, int output_n,
                              int pad_h, int pad_w,
                              const float* input_ptr, float* output_ptr, const float* kernel_ptr, float* kernel_grad_ptr)
{
  checkCUDNN(cudnnSetTensor4dDescriptor(m->inputTensor,
      CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
      input_n, input_c, input_h, input_w));

  checkCUDNN(cudnnSetTensor4dDescriptor(m->biasTensor,
      CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
      1, output_c, 1, 1));

  // Require that input_c is divisible by conv->groups
  assert(input_c % conv->groups == 0);
  printf("filterDim: kernel(%d %d) c_in(%d), c_out(%d)\n",
      conv->kernel_h, conv->kernel_w, input_c / conv->groups, output_c);
  checkCUDNN(cudnnSetFilter4dDescriptor(m->filterDesc,
      CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
      output_c, input_c / conv->groups, conv->kernel_h, conv->kernel_w));

  checkCUDNN(cudnnSetConvolution2dDescriptor(m->convDesc,
                                             pad_h,//conv->padding_h,
                                             pad_w,//conv->padding_w,
                                             conv->stride_h,
                                             conv->stride_w,
                                             1/*upscale_x*/,
                                             1/*upscale_y*/,
                                             CUDNN_CROSS_CORRELATION,
                                             CUDNN_DATA_FLOAT));
  if (conv->groups != 1) {
    checkCUDNN(cudnnSetConvolutionGroupCount(m->convDesc, conv->groups));
  }

  // enable tensor core when possible
  if (m->handle.allowTensorOpMathConversion) {
    checkCUDNN(cudnnSetConvolutionMathType(m->convDesc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
  } else {
    checkCUDNN(cudnnSetConvolutionMathType(m->convDesc, CUDNN_TENSOR_OP_MATH));
  }

  int n, c, h, w;
  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(m->convDesc,
                                                   m->inputTensor,
                                                   m->filterDesc,
                                                   &n, &c, &h, &w));
  assert(n == output_n);
  assert(c == output_c);
  assert(h == output_h);
  assert(w == output_w);

  checkCUDNN(cudnnSetTensor4dDescriptor(m->outputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        n, c, h, w));
  // select forward algorithm
  m->fwdAlgo = selectConvolutionForwardAlgorithm(m->handle.dnn, m->inputTensor, input_ptr,
                                                 m->filterDesc, kernel_ptr, m->convDesc,
                                                 m->handle.workSpace, m->handle.workSpaceSize,
                                                 m->outputTensor, output_ptr);
  // select backward filter algorithm
  m->bwdFilterAlgo = selectConvolutionBackwardFilterAlgorithm(
                         m->handle.dnn, m->inputTensor, input_ptr,
                         m->outputTensor, output_ptr,
                         m->convDesc, m->handle.workSpace, m->handle.workSpaceSize,
                         m->filterDesc, kernel_grad_ptr);
  // select backward data algorithm
  m->bwdDataAlgo = selectConvolutionBackwardDataAlgorithm(
                       m->handle.dnn, m->filterDesc, kernel_ptr,
                       m->outputTensor, output_ptr,
                       m->convDesc, m->handle.workSpace, m->handle.workSpaceSize,
                       m->inputTensor, (float*)input_ptr);
  if (m->relu) {
    checkCUDNN(cudnnSetActivationDescriptor(m->actiDesc, CUDNN_ACTIVATION_RELU,
                                            CUDNN_PROPAGATE_NAN, 0.0));
  }
}

/*static*/
void Conv2D::forward_kernel(const Conv2DMeta* m,
                            const float* input_ptr,
                            float* output_ptr,
                            const float* filter_ptr,
                            const float* bias_ptr,
                            cudaStream_t stream)
{
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(cudnnConvolutionForward(m->handle.dnn, &alpha,
                                     m->inputTensor, input_ptr,
                                     m->filterDesc, filter_ptr,
                                     m->convDesc, m->fwdAlgo,
                                     m->handle.workSpace, m->handle.workSpaceSize,
                                     &beta, m->outputTensor, output_ptr));

  // use_bias == True
  if (bias_ptr != NULL) {
    checkCUDNN(cudnnAddTensor(m->handle.dnn, &alpha, m->biasTensor,
                              bias_ptr, &alpha, m->outputTensor, output_ptr));
  }
  if (m->relu) {
    checkCUDNN(cudnnActivationForward(m->handle.dnn, m->actiDesc,
                                      &alpha, m->outputTensor, output_ptr,
                                      &beta, m->outputTensor, output_ptr));
  }
}

/*static*/
void Conv2D::forward_kernel_wrapper(const Conv2DMeta* m,
                                    const float* input_ptr,
                                    float* output_ptr,
                                    const float* filter_ptr,
                                    const float* bias_ptr)
{
  //printf("fwdAlgo(%d), bwdFilterALgo(%d), bwdDataAlgo(%d)\n", (int)m->fwdAlgo,(int) m->bwdFilterAlgo,(int) m->bwdDataAlgo);
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  
  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  Conv2D::forward_kernel(m, input_ptr, output_ptr, filter_ptr, bias_ptr, stream);
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    //print_tensor<4, float>(acc_input.ptr, acc_input.rect, "[Conv2D:forward:input]");
    //print_tensor<4, float>(acc_kernel.ptr, acc_kernel.rect, "[Conv2D:forward:kernel]");
    //print_tensor<1, float>(acc_bias.ptr, acc_bias.rect, "[Conv2D:forward:bias]");
    //print_tensor<4, float>(acc_output.ptr, acc_output.rect, "[Conv2D:forward:output]");
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("%s [Conv2D] forward time (CF) = %.2fms\n", m->op_name, elapsed);
  }
}

/*static*/
void Conv2D::backward_kernel(const Conv2DMeta* m,
                             const float* input_ptr,
                             float* input_grad_ptr,
                             const float* output_ptr,
                             float* output_grad_ptr,
                             const float* kernel_ptr,
                             float* kernel_grad_ptr,
                             float* bias_grad_ptr,
                             cudaStream_t stream)
{
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  float alpha = 1.0f;
  //float beta = 0.0f;
  if (m->relu) {
    cudnnDataType_t dataType;
    int n, c, h, w, nStride, cStride, hStride, wStride;
    checkCUDNN(cudnnGetTensor4dDescriptor(m->outputTensor, &dataType,
        &n, &c, &h, &w, &nStride, &cStride, &hStride, &wStride));
    reluBackward<<<GET_BLOCKS(n*c*h*w), CUDA_NUM_THREADS, 0, stream>>>(output_grad_ptr, output_ptr, n*c*h*w);
  }
  // Compute filter gradiant
  // NOTE: we use alpha for kernel_grad to accumulate gradients
  checkCUDNN(cudnnConvolutionBackwardFilter(m->handle.dnn, &alpha,
                                            m->inputTensor, input_ptr,
                                            m->outputTensor, output_grad_ptr,
                                            m->convDesc, m->bwdFilterAlgo,
                                            m->handle.workSpace, m->handle.workSpaceSize,
                                            &alpha, m->filterDesc, kernel_grad_ptr));
  // Compute bias gradiant
  // NOTE: we use alpha for bias_grad to accumulate gradients
  if (bias_grad_ptr != NULL) {
    checkCUDNN(cudnnConvolutionBackwardBias(m->handle.dnn, &alpha,
                                            m->outputTensor, output_grad_ptr,
                                            &alpha, m->biasTensor, bias_grad_ptr));
  }
  // Compute data gradiant
  // NOTE: we use alpha for input_grad to accumulate gradients
  if (input_grad_ptr != NULL) {
    checkCUDNN(cudnnConvolutionBackwardData(m->handle.dnn, &alpha,
                                            m->filterDesc, kernel_ptr,
                                            m->outputTensor, output_grad_ptr,
                                            m->convDesc, m->bwdDataAlgo,
                                            m->handle.workSpace, m->handle.workSpaceSize,
                                            &alpha, m->inputTensor, input_grad_ptr));
  }
}

/*static*/
void Conv2D::backward_kernel_wrapper(const Conv2DMeta* m,
                                     const float* input_ptr,
                                     float* input_grad_ptr,
                                     const float* output_ptr,
                                     float* output_grad_ptr,
                                     const float* kernel_ptr,
                                     float* kernel_grad_ptr,
                                     float* bias_grad_ptr)
{ 
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream)); 

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  Conv2D::backward_kernel(m, input_ptr, input_grad_ptr,
                          output_ptr, output_grad_ptr,
                          kernel_ptr, kernel_grad_ptr,
                          bias_grad_ptr, stream);
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("%s [Conv2D] backward time = %.2fms\n", m->op_name, elapsed);
    //print_tensor<4, float>(acc_output_grad.ptr, acc_output_grad.rect, "[Conv2D:backward:output_grad]");
    //print_tensor<4, float>(acc_kernel_grad.ptr, acc_kernel_grad.rect, "[Conv2D:backward:kernel_grad]");
    //print_tensor<1, float>(acc_bias_grad.ptr, acc_bias_grad.rect, "[Conv2D:backward:bias_grad]");
    //print_tensor<4, float>(acc_input_grad.ptr, acc_input_grad.rect, "[Conv2D:backward:input_grad]");
  }
}

cudnnConvolutionFwdAlgo_t
selectConvolutionForwardAlgorithm(cudnnHandle_t handle,
                                  const cudnnTensorDescriptor_t xDesc, const void* x,
                                  const cudnnFilterDescriptor_t wDesc, const void* w,
                                  const cudnnConvolutionDescriptor_t convDesc,
                                  void* workSpace, size_t workSpaceSize,
                                  const cudnnTensorDescriptor_t yDesc, void* y)
{
  const int reqAlgCnt = 8;
  int cnt = 0;
  cudnnConvolutionFwdAlgoPerf_t perfResults[reqAlgCnt];
  checkCUDNN(cudnnFindConvolutionForwardAlgorithmEx(
      handle, xDesc, x, wDesc, w, convDesc, yDesc, y,
      reqAlgCnt, &cnt, perfResults, workSpace, workSpaceSize));
  assert(cnt > 0);
  checkCUDNN(perfResults[0].status);
  printf("forwardAlgo(%d) time(%.2lf)\n", perfResults[0].algo, perfResults[0].time);
  return perfResults[0].algo;
}

cudnnConvolutionBwdFilterAlgo_t
selectConvolutionBackwardFilterAlgorithm(cudnnHandle_t handle,
                                         const cudnnTensorDescriptor_t xDesc, const void* x,
                                         const cudnnTensorDescriptor_t dyDesc, const void* dy,
                                         const cudnnConvolutionDescriptor_t convDesc,
                                         void* workSpace, size_t workSpaceSize,
                                         const cudnnFilterDescriptor_t dwDesc, void* dw)
{
  const int reqAlgCnt = 8;
  int cnt = 0;
  cudnnConvolutionBwdFilterAlgoPerf_t perfResults[reqAlgCnt];
  checkCUDNN(cudnnFindConvolutionBackwardFilterAlgorithmEx(
      handle, xDesc, x, dyDesc, dy, convDesc, dwDesc, dw,
      reqAlgCnt, &cnt, perfResults, workSpace, workSpaceSize));
  assert(cnt > 0);
  checkCUDNN(perfResults[0].status);
  printf("bwdFilterAlgo(%d) time(%.2lf)\n", perfResults[0].algo, perfResults[0].time);
  return perfResults[0].algo;
}

cudnnConvolutionBwdDataAlgo_t
selectConvolutionBackwardDataAlgorithm(cudnnHandle_t handle,
                                       const cudnnFilterDescriptor_t wDesc, const void* w,
                                       const cudnnTensorDescriptor_t dyDesc, const void* dy,
                                       const cudnnConvolutionDescriptor_t convDesc,
                                       void* workSpace, size_t workSpaceSize,
                                       const cudnnTensorDescriptor_t dxDesc, void* dx)
{
  const int reqAlgCnt = 8;
  int cnt = 0;
  cudnnConvolutionBwdDataAlgoPerf_t perfResults[reqAlgCnt];
  checkCUDNN(cudnnFindConvolutionBackwardDataAlgorithmEx(
      handle, wDesc, w, dyDesc, dy, convDesc, dxDesc, dx,
      reqAlgCnt, &cnt, perfResults, workSpace, workSpaceSize));
  assert(cnt > 0);
  checkCUDNN(perfResults[0].status);
  printf("bwdDataAlgo(%d) time(%.2lf)\n", perfResults[0].algo, perfResults[0].time);
  return perfResults[0].algo;
}

Conv2DMeta::Conv2DMeta(FFHandler handler)
: OpMeta(handler)
{
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&biasTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
  checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
}

// TODO: refactor it
bool Conv2D::measure_operator_cost(Simulator* sim,
                                   const ParallelConfig& pc,
                                   CostMetrics& cost_metrics) const
{
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
  checkCUDNN(cudnnSetTensor4dDescriptor(m->inputTensor, CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT, input_n, input_c, input_h, input_w));
  checkCUDNN(cudnnSetTensor4dDescriptor(m->biasTensor, CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT, 1, output_c, 1, 1));
  // require input_c is divisible by groups
  assert(input_c % groups == 0);
  checkCUDNN(cudnnSetFilter4dDescriptor(m->filterDesc, CUDNN_DATA_FLOAT,
      CUDNN_TENSOR_NCHW, output_c, input_c / groups, kernel_h, kernel_w));
  checkCUDNN(cudnnSetConvolution2dDescriptor(m->convDesc, pad_h, pad_w,
      stride_h, stride_w, 1/*dilationH*/, 1/*dilationW*/,
      CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

  checkCUDNN(cudnnSetConvolutionGroupCount(m->convDesc, groups));
  if (m->handle.allowTensorOpMathConversion) {
    checkCUDNN(cudnnSetConvolutionMathType(m->convDesc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
  } else {
    checkCUDNN(cudnnSetConvolutionMathType(m->convDesc, CUDNN_TENSOR_OP_MATH));
  }
  int n, c, h, w;
  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(m->convDesc,
      m->inputTensor, m->filterDesc, &n, &c, &h, &w));
  assert(n == output_n);
  assert(c == output_c);
  assert(h == output_h);
  assert(w == output_w);
  checkCUDNN(cudnnSetActivationDescriptor(m->actiDesc, CUDNN_ACTIVATION_RELU,
      CUDNN_NOT_PROPAGATE_NAN, 0.0));
  checkCUDNN(cudnnSetTensor4dDescriptor(m->outputTensor, CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT, n, c, h, w));
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

  // compute memory usage
  // Assume:
  //   1. all memory allocations use Simulator::allocate
  //   2. we call Simulator::free_all before measure an operator
  // Therefore, the memory usage of an operator is sim->offset
  cost_metrics.memory_requirement = (size_t)sim->offset;

  // select forward algorithm
  {
    const int reqAlgCnt = 8;
    int cnt = 0;
    cudnnConvolutionFwdAlgoPerf_t perfResults[reqAlgCnt];
    checkCUDNN(cudnnFindConvolutionForwardAlgorithmEx(
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
    cudnnConvolutionBwdFilterAlgoPerf_t perfResults[reqAlgCnt];
    checkCUDNN(cudnnFindConvolutionBackwardFilterAlgorithmEx(
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
    cudnnConvolutionBwdDataAlgoPerf_t perfResults[reqAlgCnt];
    checkCUDNN(cudnnFindConvolutionBackwardDataAlgorithmEx(
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
  return true;
}

}; // namespace FlexFlow
