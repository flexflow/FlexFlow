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

#include "device.h"
#include "kernels/allocation.h"
#include "kernels/batch_norm_kernels.h"
#include "kernels/ff_handle.h"
#include "utils/integer_conversions.h"

namespace FlexFlow {
namespace Kernels {
namespace BatchNorm {

void forward_kernel(cudaStream_t stream,
                    BatchNormPerDeviceState const &m,
                    float const *input_ptr,
                    float *output_ptr,
                    float const *scale_ptr,
                    float const *bias_ptr) {
  checkCUDNN(cudnnSetStream(m.handle.dnn, stream));

  float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(cudnnBatchNormalizationForwardTraining(m.handle.dnn,
                                                    m.mode,
                                                    &alpha,
                                                    &beta,
                                                    m.inputTensor,
                                                    input_ptr,
                                                    m.outputTensor,
                                                    output_ptr,
                                                    m.biasTensor,
                                                    scale_ptr,
                                                    bias_ptr,
                                                    1.0,
                                                    m.runningMean,
                                                    m.runningVar,
                                                    CUDNN_BN_MIN_EPSILON,
                                                    m.saveMean,
                                                    m.saveVar));
}

void backward_kernel(cudaStream_t stream,
                     BatchNormPerDeviceState const &m,
                     float const *input_ptr,
                     float *output_grad_ptr,
                     float const *output_ptr,
                     float *input_grad_ptr,
                     float const *scale_ptr,
                     float *scale_grad_ptr,
                     float *bias_grad_ptr,
                     size_t numElements) {
  checkCUDNN(cudnnSetStream(m.handle.dnn, stream));

  float alpha = 1.0f;
  if (m.relu) {
    reluBackward<<<GET_BLOCKS(numElements), CUDA_NUM_THREADS, 0, stream>>>(
        output_grad_ptr, output_ptr, numElements);
  }
  checkCUDNN(cudnnBatchNormalizationBackward(m.handle.dnn,
                                             m.mode,
                                             &alpha,
                                             &alpha,
                                             &alpha,
                                             &alpha,
                                             m.inputTensor,
                                             input_ptr,
                                             m.outputTensor,
                                             output_grad_ptr,
                                             m.inputTensor,
                                             input_grad_ptr,
                                             m.biasTensor,
                                             scale_ptr,
                                             scale_grad_ptr,
                                             bias_grad_ptr,
                                             CUDNN_BN_MIN_EPSILON,
                                             m.saveMean,
                                             m.saveVar));
}

BatchNormPerDeviceState init_kernel(PerDeviceFFHandle handle,
                                    Allocator allocator,
                                    float *runningMean,
                                    int output_n,
                                    int output_c,
                                    int output_h,
                                    int output_w,
                                    bool relu) {
  ffTensorDescriptor_t inputTensor;
  ffTensorDescriptor_t outputTensor;
  ffTensorDescriptor_t biasTensor;
  ffActivationDescriptor_t actiDesc;
  ffBatchNormMode_t mode;
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&biasTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  mode = CUDNN_BATCHNORM_SPATIAL;
#if CUDNN_VERSION >= 7000
  mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
#endif
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
  size_t totalSize = sizeof(float) * output_c * 4;
  runningMean = (float *)allocator.allocate(totalSize);
  float *runningVar = (float *)runningMean + output_c;
  float *saveMean = (float *)runningVar + output_c;
  float *saveVar = (float *)saveMean + output_c;
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  assign_kernel<<<GET_BLOCKS(output_c), CUDA_NUM_THREADS, 0, stream>>>(
      runningMean, size_t_from_int(output_c), 0.0f);
  assign_kernel<<<GET_BLOCKS(output_c), CUDA_NUM_THREADS, 0, stream>>>(
      runningVar, size_t_from_int(output_c), 0.0f);

  if (relu) {
    checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
    checkCUDNN(cudnnSetActivationDescriptor(
        actiDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
  }

  BatchNormPerDeviceState per_device_state = {handle,
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
                                              relu};

  checkCUDA(cudaStreamDestroy(stream));
  return per_device_state;
}

void cleanup_kernel(Allocator allocator,
                    ffTensorDescriptor_t inputTensor,
                    ffTensorDescriptor_t biasTensor,
                    ffTensorDescriptor_t outputTensor,
                    ffActivationDescriptor_t actiDesc,
                    bool relu,
                    float *runningMean) {
  allocator.deallocate(runningMean);
  checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
  checkCUDNN(cudnnDestroyTensorDescriptor(biasTensor));
  checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
  if (relu) {
    checkCUDNN(cudnnDestroyActivationDescriptor(actiDesc));
  }
}

} // namespace BatchNorm
} // namespace Kernels
} // namespace FlexFlow
