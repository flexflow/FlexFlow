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

#include "flexflow/ops/kernels/dropout_kernels.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

// declare Legion names
using Legion::coord_t;
using Legion::Domain;
using Legion::Memory;

DropoutMeta::DropoutMeta(FFHandler handler,
                         Dropout const *dropout,
                         Memory gpu_mem,
                         Domain const &output_domain)
    : OpMeta(handler) {
  profiling = dropout->profiling;
  rate = dropout->rate;
  seed = dropout->seed;
  input_type[0] = dropout->data_type;
  output_type[0] = dropout->data_type;
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreateDropoutDescriptor(&dropoutDesc));
  checkCUDNN(cudnnDropoutGetStatesSize(handle.dnn, &(dropoutStateSize)));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(inputTensor, output_domain));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain(outputTensor, output_domain));
  checkCUDNN(
      cudnnDropoutGetReserveSpaceSize(outputTensor, &(reserveSpaceSize)));
  {
    // allocate memory for dropoutStates and reserveSpace
    size_t totalSize = dropoutStateSize + reserveSpaceSize;
    Realm::Rect<1, coord_t> bounds(Realm::Point<1, coord_t>(0),
                                   Realm::Point<1, coord_t>(totalSize - 1));
    std::vector<size_t> field_sizes;
    field_sizes.push_back(sizeof(char));
    Realm::RegionInstance::create_instance(reserveInst,
                                           gpu_mem,
                                           bounds,
                                           field_sizes,
                                           0,
                                           Realm::ProfilingRequestSet())
        .wait();
    dropoutStates = reserveInst.pointer_untyped(0, sizeof(char));
    reserveSpace = ((char *)dropoutStates) + dropoutStateSize;
  }
  // checkCUDA(cudaMalloc(&dropoutStates, dropoutStateSize));
  // checkCUDA(cudaMalloc(&reserveSpace, reserveSpaceSize));
  checkCUDNN(cudnnSetDropoutDescriptor(dropoutDesc,
                                       handle.dnn,
                                       dropout->rate,
                                       dropoutStates,
                                       dropoutStateSize,
                                       dropout->seed));
}

DropoutMeta::~DropoutMeta(void) {
  reserveInst.destroy();
  checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
  checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
  checkCUDNN(cudnnDestroyDropoutDescriptor(dropoutDesc));
}

namespace Kernels {
namespace Dropout {

__global__ void dropout_forward_kernel(float p,
                                       long long seed,
                                       size_t num_elements,
                                       float const *input_ptr,
                                       float *output_ptr) {
  CUDA_KERNEL_LOOP(i, num_elements) {
    float scale = 1.0 / p;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, i, 0, &state);
    float rand = curand_uniform(&state);
    if (input_ptr[i] < p) {
      output_ptr[i] = 0;
    } else {
      output_ptr[i] = input_ptr[i] * scale;
    }
  }
}

__global__ void dropout_backward_kernel(float p,
                                        long long seed,
                                        size_t num_elements,
                                        float const *input_ptr,
                                        float *output_ptr) {
  CUDA_KERNEL_LOOP(i, num_elements) {
    float scale = 1.0 / p;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, i, 0, &state);
    float rand = curand_uniform(&state);
    if (input_ptr[i] < p) {
      output_ptr[i] = 0;
    } else {
      output_ptr[i] = input_ptr[i] * scale;
    }
  }
}

void forward_kernel_wrapper(DropoutMeta *m,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorW const &output) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  Internal::forward_kernel(m,
                           input.get_float_ptr(),
                           output.get_float_ptr(),
                           input.domain.get_volume(),
                           stream);
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf(" [dropout] forward time = %.2lfms\n", elapsed);
  }
}

void backward_kernel_wrapper(DropoutMeta *m,
                             GenericTensorAccessorR const &output_grad,
                             GenericTensorAccessorW const &input_grad) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  Internal::backward_kernel(m,
                            output_grad.get_float_ptr(),
                            input_grad.get_float_ptr(),
                            output_grad.domain.get_volume(),
                            stream);
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf(" [dropout] backward time = %.2lfms\n", elapsed);
  }
}

namespace Internal {

void forward_kernel(DropoutMeta *m,
                    float const *input_ptr,
                    float *output_ptr,
                    size_t num_elements,
                    cudaStream_t stream) {
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  int parallelism = num_elements;
  dropout_forward_kernel<<<GET_BLOCKS(parallelism),
                           min(CUDA_NUM_THREADS, parallelism),
                           0,
                           stream>>>(
      m->seed, m->rate, num_elements, input_ptr, output_ptr);

  // checkCUDNN(cudnnDropoutForward(m->handle.dnn,
  //                                m->dropoutDesc,
  //                                m->inputTensor,
  //                                input_ptr,
  //                                m->outputTensor,
  //                                output_ptr,
  //                                m->reserveSpace,
  //                                m->reserveSpaceSize));
}

void backward_kernel(DropoutMeta *m,
                     float const *output_grad_ptr,
                     float *input_grad_ptr,
                     size_t num_elements,
                     cudaStream_t stream) {
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
  int parallelism = num_elements;
  dropout_backward_kernel<<<GET_BLOCKS(parallelism),
                            min(CUDA_NUM_THREADS, parallelism),
                            0,
                            stream>>>(
      m->seed, m->rate, num_elements, output_grad_ptr, input_grad_ptr);

  // checkCUDNN(cudnnDropoutBackward(m->handle.dnn,
  //                                 m->dropoutDesc,
  //                                 m->outputTensor,
  //                                 output_grad_ptr,
  //                                 m->inputTensor,
  //                                 input_grad_ptr,
  //                                 m->reserveSpace,
  //                                 m->reserveSpaceSize));
}

} // namespace Internal
} // namespace Dropout
} // namespace Kernels
} // namespace FlexFlow
