#include "internal/device.h"
#include "kernels/conv_2d_kernels.h"
#include "utils/not_implemented.h"

namespace FlexFlow {
namespace Kernels {
namespace Conv2D {

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
  if (time != nullptr) {
    *time = perfResults[0].time;
  }
  return perfResults[0].algo;
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
  if (time != nullptr) {
    *time = perfResults[0].time;
  }
  return perfResults[0].algo;
}

Conv2DPerDeviceState
    gpu_init_kernel(PerDeviceFFHandle const &handle,
                    std::optional<Activation> const &activation,
                    int kernel_h,
                    int kernel_w,
                    int groups,
                    int pad_h,
                    int pad_w,
                    int stride_h,
                    int stride_w,
                    GenericTensorAccessorW const &input,
                    GenericTensorAccessorW const &output,
                    float const *filter_ptr,
                    float *filter_grad_ptr) {

  ffTensorDescriptor_t inputTensor;
  ffTensorDescriptor_t biasTensor;
  ffTensorDescriptor_t outputTensor;
  ffFilterDescriptor_t filterDesc;
  ffActivationDescriptor_t actiDesc;
  ffConvolutionDescriptor_t convDesc;
  ffConvolutionFwdAlgo_t fwdAlgo;
  ffConvolutionBwdFilterAlgo_t bwdFilterAlgo;
  ffConvolutionBwdDataAlgo_t bwdDataAlgo;

  int input_w =
      dim_at_idx(input.shape.dims, legion_dim_t{0_n}).int_from_positive_int();
  int input_h =
      dim_at_idx(input.shape.dims, legion_dim_t{1_n}).int_from_positive_int();
  int input_c =
      dim_at_idx(input.shape.dims, legion_dim_t{2_n}).int_from_positive_int();
  int input_n =
      dim_at_idx(input.shape.dims, legion_dim_t{3_n}).int_from_positive_int();

  int output_w =
      dim_at_idx(output.shape.dims, legion_dim_t{0_n}).int_from_positive_int();
  int output_h =
      dim_at_idx(output.shape.dims, legion_dim_t{1_n}).int_from_positive_int();
  int output_c =
      dim_at_idx(output.shape.dims, legion_dim_t{2_n}).int_from_positive_int();
  int output_n =
      dim_at_idx(output.shape.dims, legion_dim_t{3_n}).int_from_positive_int();

  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&biasTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
  checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));

  checkCUDNN(cudnnSetTensorDescriptorFromTensorShape(inputTensor, input.shape));

  checkCUDNN(cudnnSetTensor4dDescriptor(
      biasTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, output_c, 1, 1));

  // Require that input_c is divisible by conv->groups
  assert(input_c % groups == 0);
  checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc,
                                        CUDNN_DATA_FLOAT,
                                        CUDNN_TENSOR_NCHW,
                                        output_c,
                                        input_c / groups,
                                        kernel_h,
                                        kernel_w));

  checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                                             pad_h,
                                             pad_w,
                                             stride_h,
                                             stride_w,
                                             1 /*upscale_x*/,
                                             1 /*upscale_y*/,
                                             CUDNN_CROSS_CORRELATION,
                                             CUDNN_DATA_FLOAT));
  if (groups != 1) {
    checkCUDNN(cudnnSetConvolutionGroupCount(convDesc, groups));
  }

  // enable tensor core when possible
  if (handle.allowTensorOpMathConversion) {
    checkCUDNN(cudnnSetConvolutionMathType(
        convDesc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
  } else {
    checkCUDNN(cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH));
  }

  int n, c, h, w;
  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
      convDesc, inputTensor, filterDesc, &n, &c, &h, &w));
  assert(n == output_n);
  assert(c == output_c);
  assert(h == output_h);
  assert(w == output_w);

  checkCUDNN(cudnnSetTensor4dDescriptor(
      outputTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));

  // select forward algorithm
  fwdAlgo = selectConvolutionForwardAlgorithm(
      handle.dnn,
      inputTensor,
      static_cast<void const *>(input.get_float_ptr()),
      filterDesc,
      filter_ptr,
      convDesc,
      handle.workSpace,
      handle.workSpaceSize,
      outputTensor,
      output.get_float_ptr(),
      nullptr);

  // select backward filter algorithm
  bwdFilterAlgo = selectConvolutionBackwardFilterAlgorithm(
      handle.dnn,
      inputTensor,
      static_cast<void const *>(input.get_float_ptr()),
      outputTensor,
      output.get_float_ptr(),
      convDesc,
      handle.workSpace,
      handle.workSpaceSize,
      filterDesc,
      filter_grad_ptr,
      nullptr);

  // select backward data algorithm
  bwdDataAlgo = selectConvolutionBackwardDataAlgorithm(
      handle.dnn,
      filterDesc,
      filter_ptr,
      outputTensor,
      output.get_float_ptr(),
      convDesc,
      handle.workSpace,
      handle.workSpaceSize,
      inputTensor,
      static_cast<void *>(input.get_float_ptr()),
      nullptr);
  if (activation.has_value()) {
    checkCUDNN(cudnnSetActivationDescriptor(
        actiDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
  }

  Conv2DPerDeviceState per_device_state = Conv2DPerDeviceState{
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
  };
  return per_device_state;
}

void gpu_forward_kernel(ffStream_t stream,
                        Conv2DPerDeviceState const &m,
                        float const *input_ptr,
                        float *output_ptr,
                        float const *filter_ptr,
                        float const *bias_ptr,
                        std::optional<Activation> activation) {
  checkCUDNN(cudnnSetStream(m.handle.dnn, stream));

  float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(cudnnConvolutionForward(m.handle.dnn,
                                     &alpha,
                                     m.inputTensor,
                                     input_ptr,
                                     m.filterDesc,
                                     filter_ptr,
                                     m.convDesc,
                                     m.fwdAlgo,
                                     m.handle.workSpace,
                                     m.handle.workSpaceSize,
                                     &beta,
                                     m.outputTensor,
                                     output_ptr));

  if (bias_ptr != NULL) {
    checkCUDNN(cudnnAddTensor(m.handle.dnn,
                              &alpha,
                              m.biasTensor,
                              bias_ptr,
                              &alpha,
                              m.outputTensor,
                              output_ptr));
  }
  if (activation.has_value()) {
    checkCUDNN(cudnnActivationForward(m.handle.dnn,
                                      m.actiDesc,
                                      &alpha,
                                      m.outputTensor,
                                      output_ptr,
                                      &beta,
                                      m.outputTensor,
                                      output_ptr));
  }
}

void gpu_backward_kernel(ffStream_t stream,
                         Conv2DPerDeviceState const &m,
                         float const *output_ptr,
                         float *output_grad_ptr,
                         float const *input_ptr,
                         float *input_grad_ptr,
                         float const *filter_ptr,
                         float *filter_grad_ptr,
                         float *bias_grad_ptr,
                         std::optional<Activation> activation) {
  checkCUDNN(cudnnSetStream(m.handle.dnn, stream));

  float alpha = 1.0f;
  // float beta = 0.0f;
  if (activation.has_value()) {
    cudnnDataType_t dataType;
    int n, c, h, w, nStride, cStride, hStride, wStride;
    checkCUDNN(cudnnGetTensor4dDescriptor(m.outputTensor,
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
  // Compute filter gradiant
  // NOTE: we use alpha for kernel_grad to accumulate gradients
  checkCUDNN(cudnnConvolutionBackwardFilter(m.handle.dnn,
                                            &alpha,
                                            m.inputTensor,
                                            input_ptr,
                                            m.outputTensor,
                                            output_grad_ptr,
                                            m.convDesc,
                                            m.bwdFilterAlgo,
                                            m.handle.workSpace,
                                            m.handle.workSpaceSize,
                                            &alpha,
                                            m.filterDesc,
                                            filter_grad_ptr));
  // Compute bias gradiant
  // NOTE: we use alpha for bias_grad to accumulate gradients
  if (bias_grad_ptr != NULL) {
    checkCUDNN(cudnnConvolutionBackwardBias(m.handle.dnn,
                                            &alpha,
                                            m.outputTensor,
                                            output_grad_ptr,
                                            &alpha,
                                            m.biasTensor,
                                            bias_grad_ptr));
  }
  // Compute data gradiant
  // NOTE: we use alpha for input_grad to accumulate gradients
  if (input_grad_ptr != NULL) {
    checkCUDNN(cudnnConvolutionBackwardData(m.handle.dnn,
                                            &alpha,
                                            m.filterDesc,
                                            filter_ptr,
                                            m.outputTensor,
                                            output_grad_ptr,
                                            m.convDesc,
                                            m.bwdDataAlgo,
                                            m.handle.workSpace,
                                            m.handle.workSpaceSize,
                                            &alpha,
                                            m.inputTensor,
                                            input_grad_ptr));
  }
}

void gpu_cleanup_kernel(Conv2DPerDeviceState &per_device_state) {
  NOT_IMPLEMENTED();
}

} // namespace Conv2D
} // namespace Kernels
} // namespace FlexFlow
