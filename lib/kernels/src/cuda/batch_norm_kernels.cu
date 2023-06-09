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

#include "kernels/batch_norm_kernels.h"
#include "kernels/cuda_helper.h"

namespace FlexFlow {
namespace Kernels {
namespace BatchNorm {

/* void forward_kernel_wrapper(BatchNormPerDeviceState const *m, */
/*                                float const *input_ptr, */
/*                                float *output_ptr, */
/*                                float const *scale_ptr, */
/*                                float const *bias_ptr) { */
/*   cudaStream_t stream; */
/*    */

/*   cudaEvent_t t_start, t_end; */
/*   if (m->profiling) { */
/*     cudaEventCreate(&t_start); */
/*     cudaEventCreate(&t_end); */
/*     cudaEventRecord(t_start, stream); */
/*   } */
/*   Internal::forward_kernel(m, */
/*                  input_ptr, */
/*                  output_ptr, */
/*                  scale_ptr, */
/*                  bias_ptr); */
/*   if (m->profiling) { */
/*     cudaEventRecord(t_end, stream); */
/*     checkCUDA(cudaEventSynchronize(t_end)); */
/*     float elapsed = 0; */
/*     checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end)); */
/*     cudaEventDestroy(t_start); */
/*     cudaEventDestroy(t_end); */
/*     printf("BatchNorm forward time (BF) = %.2fms\n", elapsed); */
/*   } */
/* } */

/* void backward_kernel_wrapper(BatchNormPerDeviceState *m, */
/*                              float const *input_ptr, */
/*                              float *output_grad_ptr, */
/*                              float const *output_ptr, */
/*                              float *input_grad_ptr, */
/*                              float const *scale_ptr, */
/*                              float *scale_grad_ptr, */
/*                              float *bias_grad_ptr, */
/*                              size_t numElements) { */
/*   cudaStream_t stream; */
/*    */

/*   cudaEvent_t t_start, t_end; */
/*   if (m->profiling) { */
/*     cudaEventCreate(&t_start); */
/*     cudaEventCreate(&t_end); */
/*     cudaEventRecord(t_start, stream); */
/*   } */
/*   Internal::backward_kernel(m, */
/*                   input_ptr, */
/*                   output_grad_ptr, */
/*                   output_ptr, */
/*                   input_grad_ptr, */
/*                   scale_ptr, */
/*                   scale_grad_ptr, */
/*                   bias_grad_ptr, */
/*                   numElements); */
/*   if (m->profiling) { */
/*     cudaEventRecord(t_end, stream); */
/*     checkCUDA(cudaEventSynchronize(t_end)); */
/*     float elapsed = 0; */
/*     checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end)); */
/*     cudaEventDestroy(t_start); */
/*     cudaEventDestroy(t_end); */
/*     printf("BatchNorm backward time = %.2fms\n", elapsed); */
/*   } */

/* } */

/* namespace Internal { */

void forward_kernel(cudaStream_t stream,
                    BatchNormPerDeviceState const *m,
                    float const *input_ptr,
                    float *output_ptr,
                    float const *scale_ptr,
                    float const *bias_ptr) {
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(cudnnBatchNormalizationForwardTraining(m->handle.dnn,
                                                    m->mode,
                                                    &alpha,
                                                    &beta,
                                                    m->inputTensor,
                                                    input_ptr,
                                                    m->outputTensor,
                                                    output_ptr,
                                                    m->biasTensor,
                                                    scale_ptr,
                                                    bias_ptr,
                                                    1.0,
                                                    m->runningMean,
                                                    m->runningVar,
                                                    CUDNN_BN_MIN_EPSILON,
                                                    m->saveMean,
                                                    m->saveVar));
}

void backward_kernel(cudaStream_t stream,
                     BatchNormPerDeviceState *m,
                     float const *input_ptr,
                     float *output_grad_ptr,
                     float const *output_ptr,
                     float *input_grad_ptr,
                     float const *scale_ptr,
                     float *scale_grad_ptr,
                     float *bias_grad_ptr,
                     size_t numElements) {
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  float alpha = 1.0f;
  if (m->relu) {
    reluBackward<<<GET_BLOCKS(numElements), CUDA_NUM_THREADS, 0, stream>>>(
        output_grad_ptr, output_ptr, numElements);
  }
  checkCUDNN(cudnnBatchNormalizationBackward(m->handle.dnn,
                                             m->mode,
                                             &alpha,
                                             &alpha,
                                             &alpha,
                                             &alpha,
                                             m->inputTensor,
                                             input_ptr,
                                             m->outputTensor,
                                             output_grad_ptr,
                                             m->inputTensor,
                                             input_grad_ptr,
                                             m->biasTensor,
                                             scale_ptr,
                                             scale_grad_ptr,
                                             bias_grad_ptr,
                                             CUDNN_BN_MIN_EPSILON,
                                             m->saveMean,
                                             m->saveVar));
}

} // namespace BatchNorm
} // namespace Kernels

BatchNormPerDeviceState::BatchNormPerDeviceState(
    FFHandler handler,
    std::unique_ptr<IAllocator> allocator,
    int output_n,
    int output_c,
    int output_h,
    int output_w,
    bool relu,
    bool profiling)
    : PerDeviceOpState(handler), relu(relu), profiling(profiling),
      allocator(std::move(allocator)) {
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&biasTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  mode = CUDNN_BATCHNORM_SPATIAL;
#if CUDNN_VERSION >= 7000
  mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
#endif
  fprintf(
      stderr, "output(%d,%d,%d,%d)\n", output_n, output_c, output_h, output_w);
  checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        output_n,
                                        output_c,
                                        output_h,
                                        output_w));
  checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        output_n,
                                        output_c,
                                        output_h,
                                        output_w));
  checkCUDNN(cudnnSetTensor4dDescriptor(
      biasTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, output_c, 1, 1));
  // allocate memory for runningMean, runningVar, saveMean, saveVar
  {
    size_t totalSize = sizeof(float) * output_c * 4;
    runningMean = (float *)this->allocator->allocate(totalSize);
    runningVar = (float *)runningMean + output_c;
    saveMean = (float *)runningVar + output_c;
    saveVar = (float *)saveMean + output_c;
    cudaStream_t stream;

    assign_kernel<<<GET_BLOCKS(output_c), CUDA_NUM_THREADS, 0, stream>>>(
        runningMean, output_c, 0.0f);
    assign_kernel<<<GET_BLOCKS(output_c), CUDA_NUM_THREADS, 0, stream>>>(
        runningVar, output_c, 0.0f);
  }
  if (relu) {
    checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
    checkCUDNN(cudnnSetActivationDescriptor(
        actiDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
  }
}

BatchNormPerDeviceState::~BatchNormPerDeviceState() {
  checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
  checkCUDNN(cudnnDestroyTensorDescriptor(biasTensor));
  checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
  if (relu) {
    checkCUDNN(cudnnDestroyActivationDescriptor(actiDesc));
  }
}

} // namespace FlexFlow
