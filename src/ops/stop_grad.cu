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

#include "flexflow/ops/stop_grad.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

// declare Legion names
using Legion::coord_t;
using Legion::Domain;

template <typename T>
__global__ void stop_grad_forward_kernel(
    coord_t volume, const T *in, T *out) {
  CUDA_KERNEL_LOOP(i, volume) {
    out[i] = in[i];
  }
}

/*static*/
template <typename T>
void StopGrad::forward_kernel(StopGradMeta const *m,
                                  const T *input_ptr,
                                  T *output_ptr,
                                  size_t num_elements,
                                  cudaStream_t stream) {
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
  stop_grad_forward_kernel<<<GET_BLOCKS(num_elements),
                                  CUDA_NUM_THREADS,
                                  0,
                                  stream>>>(
      num_elements, input_ptr, output_ptr);
}

/*static*/
template <typename T>
void StopGrad::forward_kernel_wrapper(StopGradMeta const *m,
                                          const T *input_ptr,
                                          T *output_ptr,
                                          size_t num_elements) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  StopGrad::forward_kernel<T>(
      m, input_ptr, output_ptr, num_elements, stream);
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("[%s] forward time (CF) = %.2fms\n", m->op_name, elapsed);
    // print_tensor<T>(input_ptr, 32, "[EWU:forward:input]");
    // print_tensor<T>(output_ptr, 32, "[EWU:forward:output]");
  }
}

template <typename T>
__global__ void stop_grad_backward_kernel(coord_t volume,
                                              const T *output,
                                              const T *output_grad,
                                              const T *input,
                                              T *input_grad) {
  CUDA_KERNEL_LOOP(i, volume) {
    input_grad[i] = 0;
  }
}

/*static*/
template <typename T>
void StopGrad::backward_kernel(StopGradMeta const *m,
                                   const T *input_ptr,
                                   T *input_grad_ptr,
                                   const T *output_ptr,
                                   const T *output_grad_ptr,
                                   size_t num_elements,
                                   cudaStream_t stream) {
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  stop_grad_backward_kernel<T>
      <<<GET_BLOCKS(num_elements), CUDA_NUM_THREADS, 0, stream>>>(
          num_elements,
          output_ptr,
          output_grad_ptr,
          input_ptr,
          input_grad_ptr);
}

/*static*/
template <typename T>
void StopGrad::backward_kernel_wrapper(StopGradMeta const *m,
                                           const T *input_ptr,
                                           T *input_grad_ptr,
                                           const T *output_ptr,
                                           const T *output_grad_ptr,
                                           size_t num_elements) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  StopGrad::backward_kernel<T>(m,
                              input_ptr,
                              input_grad_ptr,
                              output_ptr,
                              output_grad_ptr,
                              num_elements,
                              stream);
}

StopGradMeta::StopGradMeta(FFHandler handler) : OpMeta(handler) {
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
}

template void
    StopGrad::forward_kernel_wrapper<float>(StopGradMeta const *m,
                                                float const *input_ptr,
                                                float *output_ptr,
                                                size_t num_elements);
template void
    StopGrad::forward_kernel_wrapper<double>(StopGradMeta const *m,
                                                 double const *input_ptr,
                                                 double *output_ptr,
                                                 size_t num_elements);
template void
    StopGrad::forward_kernel_wrapper<int32_t>(StopGradMeta const *m,
                                                  int32_t const *input_ptr,
                                                  int32_t *output_ptr,
                                                  size_t num_elements);
template void
    StopGrad::forward_kernel_wrapper<int64_t>(StopGradMeta const *m,
                                                  int64_t const *input_ptr,
                                                  int64_t *output_ptr,
                                                  size_t num_elements);

template void
    StopGrad::backward_kernel_wrapper<float>(StopGradMeta const *m,
                                                 float const *input_ptr,
                                                 float *input_grad_ptr,
                                                 float const *output_ptr,
                                                 float const *output_grad_ptr,
                                                 size_t num_elements);
template void
    StopGrad::backward_kernel_wrapper<double>(StopGradMeta const *m,
                                                  double const *input_ptr,
                                                  double *input_grad_ptr,
                                                  double const *output_ptr,
                                                  double const *output_grad_ptr,
                                                  size_t num_elements);
template void StopGrad::backward_kernel_wrapper<int32_t>(
    StopGradMeta const *m,
    int32_t const *input_ptr,
    int32_t *input_grad_ptr,
    int32_t const *output_ptr,
    int32_t const *output_grad_ptr,
    size_t num_elements);
template void StopGrad::backward_kernel_wrapper<int64_t>(
    StopGradMeta const *m,
    int64_t const *input_ptr,
    int64_t *input_grad_ptr,
    int64_t const *output_ptr,
    int64_t const *output_grad_ptr,
    size_t num_elements);

}; // namespace FlexFlow
