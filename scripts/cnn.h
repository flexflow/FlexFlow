#include <cudnn.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <assert.h>

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(1);                                                           \
} while(0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCUDA(status) do {                                         \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define REPEAT_TIMES 4
#define MAX_SEQ_LENGTH 40

cudnnHandle_t dnn;
cublasHandle_t blas;
void* workSpace;
size_t workSpaceSize;

float conv2DForwardTime(cudnnHandle_t handle,
                        const cudnnTensorDescriptor_t xDesc, const void* x,
                        const cudnnFilterDescriptor_t wDesc, const void* w,
                        const cudnnConvolutionDescriptor_t convDesc,
                        void* workSpace, size_t workSpaceSize,
                        const cudnnTensorDescriptor_t yDesc, void* y)
{
  const int reqAlgCnt = 8;
  int cnt = 0;
  float alpha = 1.0f, beta = 0.0f;
  cudnnConvolutionFwdAlgoPerf_t perfResults[reqAlgCnt];
  checkCUDNN(cudnnFindConvolutionForwardAlgorithmEx(
      handle, xDesc, x, wDesc, w, convDesc, yDesc, y,
      reqAlgCnt, &cnt, perfResults, workSpace, workSpaceSize));
  assert(cnt > 0);
  checkCUDNN(perfResults[0].status);
  //printf("forwardAlgo(%d) size(%zu) time(%.2lf)\n", perfResults[0].algo, perfResults[0].memory, perfResults[0].time);
  cudaEvent_t t_start, t_end;
  cudaEventCreate(&t_start);
  cudaEventCreate(&t_end);
  cudaEventRecord(t_start);
  for (int i = 0; i < REPEAT_TIMES; i++)
    checkCUDNN(cudnnConvolutionForward(handle, &alpha, xDesc, x, wDesc, w, convDesc, 
                                       perfResults[0].algo, workSpace, workSpaceSize,
                                       &beta, yDesc, y));
  cudaEventRecord(t_end);
  checkCUDA(cudaEventSynchronize(t_end));
  float elapsed = 0;
  checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  cudaEventDestroy(t_start);
  cudaEventDestroy(t_end);
  //printf("Conv2D forward time = %.2fms\n", elapsed);
  return elapsed / REPEAT_TIMES;
}

float conv2DBackwardFilterTime(cudnnHandle_t handle,
                               const cudnnTensorDescriptor_t xDesc, const void* x,
                               const cudnnTensorDescriptor_t dyDesc, const void* dy,
                               const cudnnConvolutionDescriptor_t convDesc,
                               void* workSpace, size_t workSpaceSize,
                               const cudnnFilterDescriptor_t dwDesc, void* dw)
{
  const int reqAlgCnt = 8;
  int cnt = 0;
  float alpha = 1.0f, beta = 0.0f;
  cudnnConvolutionBwdFilterAlgoPerf_t perfResults[reqAlgCnt];
  checkCUDNN(cudnnFindConvolutionBackwardFilterAlgorithmEx(
      handle, xDesc, x, dyDesc, dy, convDesc, dwDesc, dw,
      reqAlgCnt, &cnt, perfResults, workSpace, workSpaceSize));
  assert(cnt > 0);
  checkCUDNN(perfResults[0].status);
  //printf("bwdFilterAlgo(%d) size(%zu) time(%.2lf)\n", perfResults[0].algo, perfResults[0].memory, perfResults[0].time);
  cudaEvent_t t_start, t_end;
  cudaEventCreate(&t_start);
  cudaEventCreate(&t_end);
  cudaEventRecord(t_start);
  for (int i = 0; i < REPEAT_TIMES; i++)
    checkCUDNN(cudnnConvolutionBackwardFilter(handle, &alpha, xDesc, x, dyDesc, dy, convDesc,
                                              perfResults[0].algo, workSpace, workSpaceSize,
                                              &beta, dwDesc, dw));
  cudaEventRecord(t_end);
  checkCUDA(cudaEventSynchronize(t_end));
  float elapsed = 0;
  checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  cudaEventDestroy(t_start);
  cudaEventDestroy(t_end);
  //printf("Conv2D backward filter time = %.2fms\n", elapsed);
  return elapsed / REPEAT_TIMES;
}

float conv2DBackwardDataTime(cudnnHandle_t handle,
                             const cudnnFilterDescriptor_t wDesc, const void* w,
                             const cudnnTensorDescriptor_t dyDesc, const void* dy,
                             const cudnnConvolutionDescriptor_t convDesc,
                             void* workSpace, size_t workSpaceSize,
                             const cudnnTensorDescriptor_t dxDesc, void* dx)
{
  const int reqAlgCnt = 8;
  int cnt = 0;
  float alpha = 1.0f, beta = 0.0f;
  cudnnConvolutionBwdDataAlgoPerf_t perfResults[reqAlgCnt];
  checkCUDNN(cudnnFindConvolutionBackwardDataAlgorithmEx(
      handle, wDesc, w, dyDesc, dy, convDesc, dxDesc, dx,
      reqAlgCnt, &cnt, perfResults, workSpace, workSpaceSize));
  assert(cnt > 0);
  checkCUDNN(perfResults[0].status);
  //printf("bwdDataAlgo(%d) size(%zu) time(%.2lf)\n", perfResults[0].algo, perfResults[0].memory, perfResults[0].time);
  cudaEvent_t t_start, t_end;
  cudaEventCreate(&t_start);
  cudaEventCreate(&t_end);
  cudaEventRecord(t_start);
  for (int i = 0; i < REPEAT_TIMES; i++)
    checkCUDNN(cudnnConvolutionBackwardData(handle, &alpha, wDesc, w, dyDesc, dy, convDesc,
                                            perfResults[0].algo, workSpace, workSpaceSize,
                                            &beta, dxDesc, dx));
  cudaEventRecord(t_end);
  checkCUDA(cudaEventSynchronize(t_end));
  float elapsed = 0;
  checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  cudaEventDestroy(t_start);
  cudaEventDestroy(t_end);
  //printf("Conv2D backward data time = %.2fms\n", elapsed);
  return elapsed / REPEAT_TIMES;
}

float pool2DForwardTime(cudnnHandle_t handle,
                        const cudnnPoolingDescriptor_t poolDesc,
                        const cudnnTensorDescriptor_t xDesc, const void* x,
                        const cudnnTensorDescriptor_t yDesc, void* y)
{
  float alpha = 1.0f, beta = 0.0f;
  cudaEvent_t t_start, t_end;
  cudaEventCreate(&t_start);
  cudaEventCreate(&t_end);
  cudaEventRecord(t_start);
  for (int i = 0; i < REPEAT_TIMES; i++)
    checkCUDNN(cudnnPoolingForward(dnn, poolDesc, &alpha, xDesc, x,
                                   &beta, yDesc, y));
  cudaEventRecord(t_end);
  checkCUDA(cudaEventSynchronize(t_end));
  float elapsed = 0;
  checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  cudaEventDestroy(t_start);
  cudaEventDestroy(t_end);
  //printf("Conv2D backward data time = %.2fms\n", elapsed);
  return elapsed / REPEAT_TIMES;
}

float pool2DBackwardTime(cudnnHandle_t handle,
                         const cudnnPoolingDescriptor_t poolDesc,
                         const cudnnTensorDescriptor_t xDesc, const void* x,
                         const cudnnTensorDescriptor_t yDesc, const void* y,
                         const cudnnTensorDescriptor_t dxDesc, void* dx,
                         const cudnnTensorDescriptor_t dyDesc, const void* dy)
{
  float alpha = 1.0f, beta = 0.0f;
  cudaEvent_t t_start, t_end;
  cudaEventCreate(&t_start);
  cudaEventCreate(&t_end);
  cudaEventRecord(t_start);
  for (int i = 0; i < REPEAT_TIMES; i++)
    checkCUDNN(cudnnPoolingBackward(dnn, poolDesc, &alpha, yDesc, y,
                                    dyDesc, dy, xDesc, x,
                                    &beta, dxDesc, dx));
  cudaEventRecord(t_end);
  checkCUDA(cudaEventSynchronize(t_end));
  float elapsed = 0;
  checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  cudaEventDestroy(t_start);
  cudaEventDestroy(t_end);
  return elapsed / REPEAT_TIMES;
}

float init_cudnn()
{
  checkCUDNN(cudnnCreate(&dnn));
  checkCUDA(cublasCreate(&blas));
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDNN(cudnnSetStream(dnn, stream));
  checkCUDA(cublasSetStream(blas, stream));
  workSpaceSize = (size_t) 1024 * 1024 * 1024;
  checkCUDA(cudaMalloc(&workSpace, workSpaceSize));
}

float measure_conv2d_time(int batchSize, int inputSize,
                          int inputHeight, int inputWidth,
                          int outputSize,
                          int outputHeight, int outputWidth,
                          int kernelH, int kernelW,
                          int strideH, int strideW,
                          int paddingH, int paddingW)
{
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnFilterDescriptor_t filterDesc;
  cudnnConvolutionDescriptor_t convDesc;

  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

  checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                        batchSize, inputSize, inputHeight, inputWidth));
  checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                        outputSize, inputSize, kernelH, kernelW));
  checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, paddingH, paddingW, strideH, strideW,
                                             1/*upscale_x*/, 1/*upscale_y*/,
                                             CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  int n, c, h, w;
  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, inputTensor, filterDesc,
                                                   &n, &c, &h, &w));
  assert(n == batchSize);
  assert(c == outputSize);
  assert(h == outputHeight);
  assert(w == outputWidth);

  checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                        n, c, h, w));
  float *input_ptr, *filter_ptr, *output_ptr;
  size_t input_size = ((size_t)n * inputSize * inputHeight * inputWidth * sizeof(float));
  size_t filter_size = ((size_t)inputSize * outputSize * kernelH * kernelW * sizeof(float));
  size_t output_size = ((size_t)n * c * h * w * sizeof(float));
  if (input_size + filter_size + output_size > (size_t) 7 * 1024 * 1024 * 1024) {
    // No enough space
    return 1000000.0f;
  }
  checkCUDA(cudaMalloc(&input_ptr, input_size));
  checkCUDA(cudaMalloc(&filter_ptr, filter_size));
  checkCUDA(cudaMalloc(&output_ptr, output_size));
  checkCUDA(cudaDeviceSynchronize());
  float t1 = conv2DForwardTime(dnn, inputTensor, input_ptr,
                               filterDesc, filter_ptr, convDesc,
                               workSpace, workSpaceSize,
                               outputTensor, output_ptr);
  float t2 = conv2DBackwardFilterTime(dnn, inputTensor, input_ptr,
                                      outputTensor, output_ptr, convDesc,
                                      workSpace, workSpaceSize,
                                      filterDesc, filter_ptr);
  float t3 = conv2DBackwardDataTime(dnn, filterDesc, filter_ptr,
                                    outputTensor, output_ptr, convDesc,
                                    workSpace, workSpaceSize,
                                    inputTensor, input_ptr);
  checkCUDA(cudaFree(input_ptr));
  checkCUDA(cudaFree(filter_ptr));
  checkCUDA(cudaFree(output_ptr));
  checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
  checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
  checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
  checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
  printf("	Conv2D: input(%d %d %d %d) output(%d %d %d %d) k(%d %d) s(%d %d) p(%d %d) time(%.2lf) t1(%.2lf) t2+t3(%.2lf)\n",
         batchSize, inputSize, inputHeight, inputWidth,
         batchSize, outputSize, outputHeight, outputWidth,
         kernelH, kernelW, strideH, strideW, paddingH, paddingW, t1 + t2 + t3, t1, t2 + t3);
  return t1 + t2 + t3;
}

float measure_pool2d_time(int batchSize, int inputSize,
                          int inputHeight, int inputWidth,
                          int outputHeight, int outputWidth,
                          int kernelH, int kernelW,
                          int strideH, int strideW,
                          int paddingH, int paddingW,
                          bool maxpooling = true)
{
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnPoolingDescriptor_t poolDesc;

  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));

  checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                        batchSize, inputSize, inputHeight, inputWidth));
  cudnnPoolingMode_t mode;
  if (maxpooling)
    mode = CUDNN_POOLING_MAX;
  else
    mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc, mode, CUDNN_PROPAGATE_NAN,
                                         kernelH, kernelW, paddingH, paddingW, strideH, strideW));
  int n, c, h, w;
  checkCUDNN(cudnnGetPooling2dForwardOutputDim(poolDesc, inputTensor,
                                               &n, &c, &h, &w));
  assert(n == batchSize);
  assert(c == inputSize);
  assert(h == outputHeight);
  assert(w == outputWidth);

  checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                        n, c, h, w));
  float *x_ptr, *dx_ptr, *y_ptr, *dy_ptr;
  size_t input_size = ((size_t)n * inputSize * inputHeight * inputWidth * sizeof(float));
  size_t output_size = ((size_t)n * c * h * w * sizeof(float));
  if (input_size + output_size > (size_t) 7 * 1024 * 1024 * 1024) {
    // No enough space
    return 1000000.0f;
  }
  checkCUDA(cudaMalloc(&x_ptr, input_size));
  checkCUDA(cudaMalloc(&dx_ptr, input_size));
  checkCUDA(cudaMalloc(&y_ptr, output_size));
  checkCUDA(cudaMalloc(&dy_ptr, output_size));
  checkCUDA(cudaDeviceSynchronize());

  float t1 = pool2DForwardTime(dnn, poolDesc, inputTensor, x_ptr,
                               outputTensor, y_ptr);

  float t2 = pool2DBackwardTime(dnn, poolDesc, inputTensor, x_ptr,
                                outputTensor, y_ptr,
                                inputTensor, dx_ptr,
                                outputTensor, dy_ptr);

  checkCUDA(cudaFree(x_ptr));
  checkCUDA(cudaFree(dx_ptr));
  checkCUDA(cudaFree(y_ptr));
  checkCUDA(cudaFree(dy_ptr));
  checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
  checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
  checkCUDNN(cudnnDestroyPoolingDescriptor(poolDesc));
  printf("	Pool2D: input(%d %d %d %d) output(%d %d %d %d) k(%d %d) s(%d %d) p(%d %d) time(%.2lf)\n",
         batchSize, inputSize, inputHeight, inputWidth,
         batchSize, inputSize, outputHeight, outputWidth,
         kernelH, kernelW, strideH, strideW, paddingH, paddingW, t1 + t2);
  return t1 + t2;
}

float measure_linear_time(int batchSize, int inputSize, int outputSize, bool softmax)
{
  cudnnTensorDescriptor_t outputTensor;
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  cudnnStatus_t ret = cudnnSetTensor4dDescriptor(outputTensor, CUDNN_TENSOR_NCHW,
                                                 CUDNN_DATA_FLOAT, batchSize, outputSize, 1, 1);
  if (ret != CUDNN_STATUS_SUCCESS) return 1000000.0f;
  float *kernel_ptr, *input_ptr, *output_ptr, *softmax_ptr;
  float alpha = 1.0f, beta = 0.0f;
  checkCUDA(cudaMalloc(&kernel_ptr, inputSize * outputSize * sizeof(float)));
  checkCUDA(cudaMalloc(&input_ptr, batchSize * inputSize * sizeof(float)));
  checkCUDA(cudaMalloc(&output_ptr, batchSize * outputSize * sizeof(float)));
  checkCUDA(cudaMalloc(&softmax_ptr, batchSize * outputSize * sizeof(float)));
  cudaEvent_t start, stop;
  checkCUDA(cudaEventCreate(&start));
  checkCUDA(cudaEventCreate(&stop));

  // Forward Time
  checkCUDA(cudaDeviceSynchronize());
  checkCUDA(cudaEventRecord(start));
  for (int i = 0; i < REPEAT_TIMES; i++) {
    checkCUDA(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N,
                          outputSize, batchSize, inputSize,
                          &alpha, kernel_ptr, inputSize,
                          input_ptr, inputSize, &beta,
                          output_ptr, outputSize));
    checkCUDNN(cudnnSoftmaxForward(dnn, CUDNN_SOFTMAX_ACCURATE,
                                   CUDNN_SOFTMAX_MODE_CHANNEL,
                                   &alpha, outputTensor, output_ptr,
                                   &beta, outputTensor, softmax_ptr));
  }
  checkCUDA(cudaEventRecord(stop));
  checkCUDA(cudaEventSynchronize(stop));
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  float t1 = milliseconds / REPEAT_TIMES;

  // Backward Time
  checkCUDA(cudaDeviceSynchronize());
  checkCUDA(cudaEventRecord(start));
  for (int i = 0; i < REPEAT_TIMES; i++) {
    checkCUDA(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T,
                          inputSize, outputSize, batchSize,
                          &alpha, input_ptr, inputSize,
                          output_ptr, outputSize,
                          &beta, kernel_ptr, inputSize));
    checkCUDA(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
                          inputSize, batchSize, outputSize,
                          &alpha, kernel_ptr, inputSize,
                          output_ptr, outputSize,
                          &beta, input_ptr, inputSize));
  }
  checkCUDA(cudaEventRecord(stop));
  checkCUDA(cudaEventSynchronize(stop));
  cudaEventElapsedTime(&milliseconds, start, stop);
  float t2 = milliseconds / REPEAT_TIMES;

  checkCUDA(cudaFree(kernel_ptr));
  checkCUDA(cudaFree(input_ptr));
  checkCUDA(cudaFree(output_ptr));
  checkCUDA(cudaFree(softmax_ptr));
  checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));

  printf("Linear: input(%d %d) output(%d %d) t(%.2lf)\n", batchSize, inputSize,
         batchSize, outputSize, t1 + t2);
  return t1 + t2;
}

float measure_lstm_time(int numLayers, int seqLength, int batchSize, int inputSize, int outputSize)
{
  cudnnRNNDescriptor_t rnnDesc;
  cudnnDropoutDescriptor_t dropoutDesc;
  cudnnTensorDescriptor_t xDescs[MAX_SEQ_LENGTH], yDescs[MAX_SEQ_LENGTH];
  cudnnTensorDescriptor_t cxDesc, hxDesc, cyDesc, hyDesc;
  float *xData, *yData, *cxData, *cyData, *hxData, *hyData;
  cudnnFilterDescriptor_t wDesc;

  float dropout = 0.2f;
  checkCUDNN(cudnnCreateRNNDescriptor(&rnnDesc));
  checkCUDNN(cudnnCreateDropoutDescriptor(&dropoutDesc));
  size_t dropoutSize;
  void *dropoutStates;
  checkCUDNN(cudnnDropoutGetStatesSize(dnn, &dropoutSize));
  checkCUDA(cudaMalloc(&dropoutStates, dropoutSize));
  checkCUDNN(cudnnSetDropoutDescriptor(dropoutDesc, dnn, dropout, dropoutStates, dropoutSize, 10));
  checkCUDNN(cudnnSetRNNDescriptor_v5(rnnDesc, outputSize, numLayers, dropoutDesc,
                                      CUDNN_LINEAR_INPUT, CUDNN_UNIDIRECTIONAL, CUDNN_LSTM,
                                      CUDNN_DATA_FLOAT));
  // Create input descriptors
  for (int i = 0; i < seqLength; i++) {
    checkCUDNN(cudnnCreateTensorDescriptor(&xDescs[i]));
    int dims[] = {batchSize, inputSize, 1};
    int strides[] = {dims[1] * dims[2], dims[2], 1};
    checkCUDNN(cudnnSetTensorNdDescriptor(xDescs[i], CUDNN_DATA_FLOAT, 3, dims, strides));
  }
  checkCUDA(cudaMalloc(&xData, batchSize * outputSize * seqLength * sizeof(float)));
  // Workspace size
  size_t myWSS;
  checkCUDNN(cudnnGetRNNWorkspaceSize(dnn, rnnDesc, seqLength, xDescs, &myWSS));
  // ReserveSpace size
  size_t reserveSpaceSize;
  void* reserveSpace;
  checkCUDNN(cudnnGetRNNTrainingReserveSize(dnn, rnnDesc, seqLength, xDescs, &reserveSpaceSize));
  checkCUDA(cudaMalloc(&reserveSpace, reserveSpaceSize));
  //printf("reserveSpaceSize = %zu\n", reserveSpaceSize);
  // Params Size
  size_t paramsSize;
  void* paramsSpace;
  checkCUDNN(cudnnGetRNNParamsSize(dnn, rnnDesc, xDescs[0], &paramsSize, CUDNN_DATA_FLOAT));
  checkCUDA(cudaMalloc(&paramsSpace, paramsSize));
  //printf("paramsSize = %zu\n", paramsSize);
  // Create weight filter
  {
    int dims[] = {(int)paramsSize, 1, 1};
    checkCUDNN(cudnnCreateFilterDescriptor(&wDesc));
    checkCUDNN(cudnnSetFilterNdDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dims));
  }
  // Initialize hx, cx, hy, cy
  {
    checkCUDNN(cudnnCreateTensorDescriptor(&hxDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&cxDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&hyDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&cyDesc));
    int dims[] = {numLayers, batchSize, outputSize};
    int strides[] = {dims[1] * dims[2], dims[2], 1};
    checkCUDNN(cudnnSetTensorNdDescriptor(hxDesc, CUDNN_DATA_FLOAT, 3, dims, strides));
    checkCUDNN(cudnnSetTensorNdDescriptor(cxDesc, CUDNN_DATA_FLOAT, 3, dims, strides));
    checkCUDNN(cudnnSetTensorNdDescriptor(hyDesc, CUDNN_DATA_FLOAT, 3, dims, strides));
    checkCUDNN(cudnnSetTensorNdDescriptor(cyDesc, CUDNN_DATA_FLOAT, 3, dims, strides));
    checkCUDA(cudaMalloc(&hxData, dims[0] * dims[1] * dims[2] * sizeof(float)));
    checkCUDA(cudaMalloc(&cxData, dims[0] * dims[1] * dims[2] * sizeof(float)));
    checkCUDA(cudaMalloc(&hyData, dims[0] * dims[1] * dims[2] * sizeof(float)));
    checkCUDA(cudaMalloc(&cyData, dims[0] * dims[1] * dims[2] * sizeof(float)));
  }
  // Initialize yDescs
  for (int i = 0; i < seqLength; i++) {
    checkCUDNN(cudnnCreateTensorDescriptor(&yDescs[i]));
    int dims[] = {batchSize, outputSize, 1};
    int strides[] = {dims[1] * dims[2], dims[2], 1};
    checkCUDNN(cudnnSetTensorNdDescriptor(yDescs[i], CUDNN_DATA_FLOAT, 3, dims, strides));
  }
  checkCUDA(cudaMalloc(&yData, batchSize * outputSize * seqLength * sizeof(float)));

  cudaEvent_t start, stop;
  checkCUDA(cudaEventCreate(&start));
  checkCUDA(cudaEventCreate(&stop));
  checkCUDA(cudaDeviceSynchronize());
  float elapsed = 0;

  for (int i = 0; i < 2*REPEAT_TIMES; i++) {
    // Use the first REPEAT_TIMES to warm up
    if (i==REPEAT_TIMES) {
      checkCUDA(cudaDeviceSynchronize());
      checkCUDA(cudaEventRecord(start));
    }
    checkCUDNN(cudnnRNNForwardTraining(dnn, rnnDesc, seqLength, xDescs, xData, hxDesc, hxData, cxDesc, cxData, wDesc, paramsSpace,
                                       yDescs, yData, hyDesc, hyData, cyDesc, cyData, workSpace, workSpaceSize, reserveSpace, reserveSpaceSize));
  }
  checkCUDA(cudaEventRecord(stop));
  checkCUDA(cudaEventSynchronize(stop));
  cudaEventElapsedTime(&elapsed, start, stop);
  float t1 = elapsed / REPEAT_TIMES;
  // Backward Time
  checkCUDA(cudaDeviceSynchronize());
  for (int i = 0; i < 2*REPEAT_TIMES; i++) {
    if (i == REPEAT_TIMES) {
      checkCUDA(cudaDeviceSynchronize());
      checkCUDA(cudaEventRecord(start));
    }
    checkCUDNN(cudnnRNNBackwardData(dnn, rnnDesc, seqLength, yDescs, yData, yDescs, yData, hyDesc, hyData, cyDesc, cyData, wDesc, paramsSpace,
                                    hxDesc, hxData, cxDesc, cxData, xDescs, xData, hxDesc, hxData, cxDesc, cxData, workSpace, workSpaceSize, reserveSpace, reserveSpaceSize));
    checkCUDNN(cudnnRNNBackwardWeights(dnn, rnnDesc, seqLength, xDescs, xData, hxDesc, hxData, yDescs, yData, workSpace, workSpaceSize, wDesc, paramsSpace,
                                       reserveSpace, reserveSpaceSize));
  }
  checkCUDA(cudaEventRecord(stop));
  checkCUDA(cudaEventSynchronize(stop));
  cudaEventElapsedTime(&elapsed, start, stop);
  float t2 = elapsed / REPEAT_TIMES;
  checkCUDA(cudaEventDestroy(start));
  checkCUDA(cudaEventDestroy(stop));
  
  checkCUDNN(cudnnDestroyRNNDescriptor(rnnDesc));
  checkCUDNN(cudnnDestroyDropoutDescriptor(dropoutDesc));
  checkCUDA(cudaFree(dropoutStates));
  for (int i = 0; i < seqLength; i++) {
    checkCUDNN(cudnnDestroyTensorDescriptor(xDescs[i]));
    checkCUDNN(cudnnDestroyTensorDescriptor(yDescs[i]));
  }
  checkCUDA(cudaFree(xData));
  checkCUDA(cudaFree(yData));
  checkCUDA(cudaFree(reserveSpace));
  checkCUDA(cudaFree(paramsSpace));
  checkCUDNN(cudnnDestroyTensorDescriptor(hxDesc));
  checkCUDNN(cudnnDestroyTensorDescriptor(cxDesc));
  checkCUDNN(cudnnDestroyTensorDescriptor(hyDesc));
  checkCUDNN(cudnnDestroyTensorDescriptor(cyDesc));
  checkCUDA(cudaFree(hxData));
  checkCUDA(cudaFree(cxData));
  checkCUDA(cudaFree(hyData));
  checkCUDA(cudaFree(cyData));
  printf("	LSTM: batch(%d) input(%d) output(%d) t1(%.2lf) t2(%.2lf)\n", batchSize, inputSize, outputSize, t1, t2);
  return t1 + t2;
}
