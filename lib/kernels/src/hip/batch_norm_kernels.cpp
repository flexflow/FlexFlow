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
#include "kernels/hip_helper.h"
#include <hip/hip_runtime.h>

namespace FlexFlow {

// declare Legion names
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::Machine;
using Legion::Memory;
using Legion::PhysicalRegion;
using Legion::Rect;
using Legion::Runtime;
using Legion::Task;

#define MIOPEN_BN_MIN_EPSILON 0.001

namespace Kernels {
namespace BatchNorm {

void forward_kernel(hipStream_t stream, BatchNormPerDeviceState *m,
                    float const *input_ptr, float *output_ptr,
                    float const *scale_ptr, float const *bias_ptr) {

  checkCUDNN(miopenSetStream(m->handle.dnn, stream));

  float alpha = 1.0f, beta = 0.0f;
  // coord_t numChannels = m->numChannels;
  checkCUDNN(miopenBatchNormalizationForwardTraining(
      m->handle.dnn, m->mode, &alpha, &beta, m->inputTensor, input_ptr,
      m->outputTensor, output_ptr, m->biasTensor,
      static_cast<void *>(const_cast<float *>(scale_ptr)),
      static_cast<void *>(const_cast<float *>(bias_ptr)), 1.0, m->runningMean,
      m->runningVar, MIOPEN_BN_MIN_EPSILON, m->saveMean, m->saveVar));
}

void backward_kernel(hipStream_t stream, BatchNormPerDeviceState *m,
                     float const *input_ptr, float *output_grad_ptr,
                     float const *output_ptr, float *input_grad_ptr,
                     float const *scale_ptr, float *scale_grad_ptr,
                     float *bias_grad_ptr, size_t numElements) {

  checkCUDNN(miopenSetStream(m->handle.dnn, stream));

  float alpha = 1.0f;
  if (m->relu) {
    hipLaunchKernelGGL(reluBackward, GET_BLOCKS(numElements), CUDA_NUM_THREADS,
                       0, stream, output_grad_ptr, output_ptr, numElements);
  }
  checkCUDNN(miopenBatchNormalizationBackward(
      m->handle.dnn, m->mode, &alpha, &alpha, &alpha, &alpha, m->inputTensor,
      input_ptr, m->outputTensor, output_grad_ptr, m->inputTensor,
      input_grad_ptr, m->biasTensor, scale_ptr, scale_grad_ptr, bias_grad_ptr,
      MIOPEN_BN_MIN_EPSILON, m->saveMean, m->saveVar));
}

} // namespace BatchNorm
} // namespace Kernels

BatchNormPerDeviceState::BatchNormPerDeviceState(FFHandler handler,
                                                 BatchNorm const *bn,
                                                 Memory gpu_mem, int output_n,
                                                 int output_c, int output_h,
                                                 int output_w)
    : PerDeviceOpState(handler) {
  checkCUDNN(miopenCreateTensorDescriptor(&inputTensor));
  checkCUDNN(miopenCreateTensorDescriptor(&biasTensor));
  checkCUDNN(miopenCreateTensorDescriptor(&outputTensor));
  relu = bn->relu;
  profiling = bn->profiling;
  mode = miopenBNSpatial;
  // #if HIPDNN_VERSION >= 7000
  //   mode = HIPDNN_BATCHNORM_SPATIAL_PERSISTENT;
  // #endif
  fprintf(stderr, "output(%d,%d,%d,%d)\n", output_n, output_c, output_h,
          output_w);
  checkCUDNN(miopenSet4dTensorDescriptor(inputTensor, miopenFloat, output_n,
                                         output_c, output_h, output_w));
  checkCUDNN(miopenSet4dTensorDescriptor(outputTensor, miopenFloat, output_n,
                                         output_c, output_h, output_w));
  checkCUDNN(
      miopenSet4dTensorDescriptor(biasTensor, miopenFloat, 1, output_c, 1, 1));
  // allocate memory for runningMean, runningVar, saveMean, saveVar
  {
    size_t totalSize = sizeof(float) * output_c * 4;
    // Realm::Rect<1, coord_t> bounds(Realm::Point<1, coord_t>(0),
    //                                Realm::Point<1, coord_t>(totalSize - 1));
    // std::vector<size_t> field_sizes;
    // field_sizes.push_back(sizeof(char));
    // Realm::RegionInstance::create_instance(reserveInst,
    //                                        gpu_mem,
    //                                        bounds,
    //                                        field_sizes,
    //                                        0,
    //                                        Realm::ProfilingRequestSet())
    //     .wait();
    runningMean = (float *)this->allocator->allocate(totalSize);
    runningVar = (float *)runningMean + output_c;
    saveMean = (float *)runningVar + output_c;
    saveVar = (float *)saveMean + output_c;
    hipStream_t stream;

    hipLaunchKernelGGL(assign_kernel, GET_BLOCKS(output_c), CUDA_NUM_THREADS, 0,
                       stream, runningMean, output_c, 0.0f);
    hipLaunchKernelGGL(assign_kernel, GET_BLOCKS(output_c), CUDA_NUM_THREADS, 0,
                       stream, runningVar, output_c, 0.0f);
  }
  if (relu) {
    checkCUDNN(miopenCreateActivationDescriptor(&actiDesc));
    checkCUDNN(miopenSetActivationDescriptor(actiDesc, miopenActivationRELU,
                                             0.0, 0.0, 0.0));
  }
}

BatchNormPerDeviceState::~BatchNormPerDeviceState(void) {
  reserveInst.destroy();
  checkCUDNN(miopenDestroyTensorDescriptor(inputTensor));
  checkCUDNN(miopenDestroyTensorDescriptor(biasTensor));
  checkCUDNN(miopenDestroyTensorDescriptor(outputTensor));
  if (relu) {
    checkCUDNN(miopenDestroyActivationDescriptor(actiDesc));
  }
}

} // namespace FlexFlow
