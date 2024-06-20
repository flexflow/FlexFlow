#include "flexflow/ops/conv_2d.h"
#include "flexflow/ops/kernels/conv_2d_kernels.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

Conv2DMeta::Conv2DMeta(FFHandler handler, Conv2D const *conv)
    : OpMeta(handler, conv) {
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&biasTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
  checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
}

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
                 float *forward_time,
                 float *backward_time) {
  checkCUDNN(cudnnSetTensor4dDescriptor(m->inputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        input_n,
                                        input_c,
                                        input_h,
                                        input_w));

  checkCUDNN(cudnnSetTensor4dDescriptor(
      m->biasTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, output_c, 1, 1));

  // Require that input_c is divisible by conv->groups
  assert(input_c % groups == 0);
  printf("filterDim: kernel(%d %d) c_in(%d), c_out(%d)\n",
         kernel_h,
         kernel_w,
         input_c / groups,
         output_c);
  checkCUDNN(cudnnSetFilter4dDescriptor(m->filterDesc,
                                        CUDNN_DATA_FLOAT,
                                        CUDNN_TENSOR_NCHW,
                                        output_c,
                                        input_c / groups,
                                        kernel_h,
                                        kernel_w));

  checkCUDNN(cudnnSetConvolution2dDescriptor(m->convDesc,
                                             pad_h, // conv->padding_h,
                                             pad_w, // conv->padding_w,
                                             stride_h,
                                             stride_w,
                                             1 /*upscale_x*/,
                                             1 /*upscale_y*/,
                                             CUDNN_CROSS_CORRELATION,
                                             CUDNN_DATA_FLOAT));
  if (groups != 1) {
    checkCUDNN(cudnnSetConvolutionGroupCount(m->convDesc, groups));
  }

  // enable tensor core when possible
  if (m->handle.allowTensorOpMathConversion) {
    checkCUDNN(cudnnSetConvolutionMathType(
        m->convDesc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
  } else {
    checkCUDNN(cudnnSetConvolutionMathType(m->convDesc, CUDNN_TENSOR_OP_MATH));
  }

  int n, c, h, w;
  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
      m->convDesc, m->inputTensor, m->filterDesc, &n, &c, &h, &w));
  assert(n == output_n);
  assert(c == output_c);
  assert(h == output_h);
  assert(w == output_w);

  checkCUDNN(cudnnSetTensor4dDescriptor(
      m->outputTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));

  float time;
  // select forward algorithm
  m->fwdAlgo =
      Internal::selectConvolutionForwardAlgorithm(m->handle.dnn,
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
  m->bwdFilterAlgo = Internal::selectConvolutionBackwardFilterAlgorithm(
      m->handle.dnn,
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
      Internal::selectConvolutionBackwardDataAlgorithm(m->handle.dnn,
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
    checkCUDNN(cudnnSetActivationDescriptor(
        m->actiDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
  }
}

void forward_kernel_wrapper(Conv2DMeta const *m,
                            float const *input_ptr,
                            float *output_ptr,
                            float const *filter_ptr,
                            float const *bias_ptr) {
  // printf("fwdAlgo(%d), bwdFilterALgo(%d), bwdDataAlgo(%d)\n",
  // (int)m->fwdAlgo,(int) m->bwdFilterAlgo,(int) m->bwdDataAlgo);
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  Internal::forward_kernel(
      m, input_ptr, output_ptr, filter_ptr, bias_ptr, stream);
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    print_tensor<float>(input_ptr, 16, "[Conv2D:forward:input]");
    print_tensor<float>(filter_ptr, 16, "[Conv2D:forward:kernel]");
    print_tensor<float>(bias_ptr, 16, "[Conv2D:forward:bias]");
    print_tensor<float>(output_ptr, 16, "[Conv2D:forward:output]");
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("%s [Conv2D] forward time (CF) = %.2fms\n", m->op_name, elapsed);
  }
}

void backward_kernel_wrapper(Conv2DMeta const *m,
                             float const *input_ptr,
                             float *input_grad_ptr,
                             float const *output_ptr,
                             float *output_grad_ptr,
                             float const *kernel_ptr,
                             float *kernel_grad_ptr,
                             float *bias_grad_ptr) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  Internal::backward_kernel(m,
                            input_ptr,
                            input_grad_ptr,
                            output_ptr,
                            output_grad_ptr,
                            kernel_ptr,
                            kernel_grad_ptr,
                            bias_grad_ptr,
                            stream);
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
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

namespace Internal {

void forward_kernel(Conv2DMeta const *m,
                    float const *input_ptr,
                    float *output_ptr,
                    float const *filter_ptr,
                    float const *bias_ptr,
                    cudaStream_t stream) {
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(cudnnConvolutionForward(m->handle.dnn,
                                     &alpha,
                                     m->inputTensor,
                                     input_ptr,
                                     m->filterDesc,
                                     filter_ptr,
                                     m->convDesc,
                                     m->fwdAlgo,
                                     m->handle.workSpace,
                                     m->handle.workSpaceSize,
                                     &beta,
                                     m->outputTensor,
                                     output_ptr));

  // use_bias == True
  if (bias_ptr != NULL) {
    checkCUDNN(cudnnAddTensor(m->handle.dnn,
                              &alpha,
                              m->biasTensor,
                              bias_ptr,
                              &alpha,
                              m->outputTensor,
                              output_ptr));
  }
  if (m->relu) {
    checkCUDNN(cudnnActivationForward(m->handle.dnn,
                                      m->actiDesc,
                                      &alpha,
                                      m->outputTensor,
                                      output_ptr,
                                      &beta,
                                      m->outputTensor,
                                      output_ptr));
  }
}

void backward_kernel(Conv2DMeta const *m,
                     float const *input_ptr,
                     float *input_grad_ptr,
                     float const *output_ptr,
                     float *output_grad_ptr,
                     float const *kernel_ptr,
                     float *kernel_grad_ptr,
                     float *bias_grad_ptr,
                     cudaStream_t stream) {
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  float alpha = 1.0f;
  // float beta = 0.0f;
  if (m->relu) {
    cudnnDataType_t dataType;
    int n, c, h, w, nStride, cStride, hStride, wStride;
    checkCUDNN(cudnnGetTensor4dDescriptor(m->outputTensor,
                                          &dataType,
                                          &n,
                                          &c,
                                          &h,
                                          &w,
                                          &nStride,
                                          &cStride,
                                          &hStride,
                                          &wStride));
    reluBackward<<<GET_BLOCKS(n * c * h * w), CUDA_NUM_THREADS, 0, stream>>>(
        output_grad_ptr, output_ptr, n * c * h * w);
  }
  // Compute filter gradient
  // NOTE: we use alpha for kernel_grad to accumulate gradients
  checkCUDNN(cudnnConvolutionBackwardFilter(m->handle.dnn,
                                            &alpha,
                                            m->inputTensor,
                                            input_ptr,
                                            m->outputTensor,
                                            output_grad_ptr,
                                            m->convDesc,
                                            m->bwdFilterAlgo,
                                            m->handle.workSpace,
                                            m->handle.workSpaceSize,
                                            &alpha,
                                            m->filterDesc,
                                            kernel_grad_ptr));
  // Compute bias gradient
  // NOTE: we use alpha for bias_grad to accumulate gradients
  if (bias_grad_ptr != NULL) {
    checkCUDNN(cudnnConvolutionBackwardBias(m->handle.dnn,
                                            &alpha,
                                            m->outputTensor,
                                            output_grad_ptr,
                                            &alpha,
                                            m->biasTensor,
                                            bias_grad_ptr));
  }
  // Compute data gradient
  // NOTE: we use alpha for input_grad to accumulate gradients
  if (input_grad_ptr != NULL) {
    checkCUDNN(cudnnConvolutionBackwardData(m->handle.dnn,
                                            &alpha,
                                            m->filterDesc,
                                            kernel_ptr,
                                            m->outputTensor,
                                            output_grad_ptr,
                                            m->convDesc,
                                            m->bwdDataAlgo,
                                            m->handle.workSpace,
                                            m->handle.workSpaceSize,
                                            &alpha,
                                            m->inputTensor,
                                            input_grad_ptr));
  }
}

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
    float *time) {
  int const reqAlgCnt = 8;
  int cnt = 0;
  cudnnConvolutionFwdAlgoPerf_t perfResults[reqAlgCnt];
  checkCUDNN(cudnnFindConvolutionForwardAlgorithmEx(handle,
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
                                                    workSpaceSize));
  assert(cnt > 0);
  checkCUDNN(perfResults[0].status);
  printf("forwardAlgo(%d) time(%.2lf)\n",
         perfResults[0].algo,
         perfResults[0].time);
  if (time != nullptr) {
    *time = perfResults[0].time;
  }
  return perfResults[0].algo;
}

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
    float *time) {
  int const reqAlgCnt = 8;
  int cnt = 0;
  cudnnConvolutionBwdDataAlgoPerf_t perfResults[reqAlgCnt];
  checkCUDNN(cudnnFindConvolutionBackwardDataAlgorithmEx(handle,
                                                         wDesc,
                                                         w,
                                                         dyDesc,
                                                         dy,
                                                         convDesc,
                                                         dxDesc,
                                                         dx,
                                                         reqAlgCnt,
                                                         &cnt,
                                                         perfResults,
                                                         workSpace,
                                                         workSpaceSize));
  assert(cnt > 0);
  checkCUDNN(perfResults[0].status);
  printf("bwdDataAlgo(%d) time(%.2lf)\n",
         perfResults[0].algo,
         perfResults[0].time);
  if (time != nullptr) {
    *time = perfResults[0].time;
  }
  return perfResults[0].algo;
}

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
    float *time) {
  int const reqAlgCnt = 8;
  int cnt = 0;
  cudnnConvolutionBwdFilterAlgoPerf_t perfResults[reqAlgCnt];
  checkCUDNN(cudnnFindConvolutionBackwardFilterAlgorithmEx(handle,
                                                           xDesc,
                                                           x,
                                                           dyDesc,
                                                           dy,
                                                           convDesc,
                                                           dwDesc,
                                                           dw,
                                                           reqAlgCnt,
                                                           &cnt,
                                                           perfResults,
                                                           workSpace,
                                                           workSpaceSize));
  assert(cnt > 0);
  checkCUDNN(perfResults[0].status);
  printf("bwdFilterAlgo(%d) time(%.2lf)\n",
         perfResults[0].algo,
         perfResults[0].time);
  if (time != nullptr) {
    *time = perfResults[0].time;
  }
  return perfResults[0].algo;
}

} // namespace Internal
} // namespace Conv2D
} // namespace Kernels
} // namespace FlexFlow
