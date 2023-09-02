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

#include "flexflow/ops/kernels/cast_kernels.h"
#include "flexflow/utils/hip_helper.h"
#include <hip/hip_runtime.h>

namespace FlexFlow {

CastMeta::CastMeta(FFHandler handle) : OpMeta(handle) {}

namespace Kernels {
namespace Cast {

template <typename IDT, typename ODT>
void forward_kernel_wrapper(CastMeta const *m,
                            IDT const *input_ptr,
                            ODT *output_ptr,
                            size_t volume) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  hipEvent_t t_start, t_end;
  if (m->profiling) {
    checkCUDA(hipEventCreate(&t_start));
    checkCUDA(hipEventCreate(&t_end));
    checkCUDA(hipEventRecord(t_start, stream));
  }

  Internal::forward_kernel<IDT, ODT>(input_ptr, output_ptr, volume, stream);
  if (m->profiling) {
    checkCUDA(hipEventRecord(t_end, stream));
    checkCUDA(hipEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(hipEventElapsedTime(&elapsed, t_start, t_end));
    checkCUDA(hipEventDestroy(t_start));
    checkCUDA(hipEventDestroy(t_end));
    printf("[%s] forward time (CF) = %.2fms\n", "Cast", elapsed);
    // print_tensor<IDT>(input_ptr, 32, "[Cast:forward:input]");
    // print_tensor<ODT>(output_ptr, 32, "[Cast:forward:output]");
  }
}

template void forward_kernel_wrapper<float, float>(CastMeta const *m,
                                                   float const *input_ptr,
                                                   float *output_ptr,
                                                   size_t volume);
template void forward_kernel_wrapper<float, double>(CastMeta const *m,
                                                    float const *input_ptr,
                                                    double *output_ptr,
                                                    size_t volume);
template void forward_kernel_wrapper<float, int32_t>(CastMeta const *m,
                                                     float const *input_ptr,
                                                     int32_t *output_ptr,
                                                     size_t volume);
template void forward_kernel_wrapper<float, int64_t>(CastMeta const *m,
                                                     float const *input_ptr,
                                                     int64_t *output_ptr,
                                                     size_t volume);

template void forward_kernel_wrapper<double, float>(CastMeta const *m,
                                                    double const *input_ptr,
                                                    float *output_ptr,
                                                    size_t volume);
template void forward_kernel_wrapper<double, double>(CastMeta const *m,
                                                     double const *input_ptr,
                                                     double *output_ptr,
                                                     size_t volume);
template void forward_kernel_wrapper<double, int32_t>(CastMeta const *m,
                                                      double const *input_ptr,
                                                      int32_t *output_ptr,
                                                      size_t volume);
template void forward_kernel_wrapper<double, int64_t>(CastMeta const *m,
                                                      double const *input_ptr,
                                                      int64_t *output_ptr,
                                                      size_t volume);

template void forward_kernel_wrapper<int32_t, float>(CastMeta const *m,
                                                     int32_t const *input_ptr,
                                                     float *output_ptr,
                                                     size_t volume);
template void forward_kernel_wrapper<int32_t, double>(CastMeta const *m,
                                                      int32_t const *input_ptr,
                                                      double *output_ptr,
                                                      size_t volume);
template void forward_kernel_wrapper<int32_t, int32_t>(CastMeta const *m,
                                                       int32_t const *input_ptr,
                                                       int32_t *output_ptr,
                                                       size_t volume);
template void forward_kernel_wrapper<int32_t, int64_t>(CastMeta const *m,
                                                       int32_t const *input_ptr,
                                                       int64_t *output_ptr,
                                                       size_t volume);

template void forward_kernel_wrapper<int64_t, float>(CastMeta const *m,
                                                     int64_t const *input_ptr,
                                                     float *output_ptr,
                                                     size_t volume);
template void forward_kernel_wrapper<int64_t, double>(CastMeta const *m,
                                                      int64_t const *input_ptr,
                                                      double *output_ptr,
                                                      size_t volume);
template void forward_kernel_wrapper<int64_t, int32_t>(CastMeta const *m,
                                                       int64_t const *input_ptr,
                                                       int32_t *output_ptr,
                                                       size_t volume);
template void forward_kernel_wrapper<int64_t, int64_t>(CastMeta const *m,
                                                       int64_t const *input_ptr,
                                                       int64_t *output_ptr,
                                                       size_t volume);

template <typename IDT, typename ODT>
void backward_kernel_wrapper(IDT const *src_ptr, ODT *dst_ptr, size_t volume) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  Internal::backward_kernel<IDT, ODT>(src_ptr, dst_ptr, volume, stream);
}

template void backward_kernel_wrapper<float, float>(float const *src_ptr,
                                                    float *dst_ptr,
                                                    size_t volume);
template void backward_kernel_wrapper<float, double>(float const *src_ptr,
                                                     double *dst_ptr,
                                                     size_t volume);
template void backward_kernel_wrapper<float, int32_t>(float const *src_ptr,
                                                      int32_t *dst_ptr,
                                                      size_t volume);
template void backward_kernel_wrapper<float, int64_t>(float const *src_ptr,
                                                      int64_t *dst_ptr,
                                                      size_t volume);

template void backward_kernel_wrapper<double, float>(double const *src_ptr,
                                                     float *dst_ptr,
                                                     size_t volume);
template void backward_kernel_wrapper<double, double>(double const *src_ptr,
                                                      double *dst_ptr,
                                                      size_t volume);
template void backward_kernel_wrapper<double, int32_t>(double const *src_ptr,
                                                       int32_t *dst_ptr,
                                                       size_t volume);
template void backward_kernel_wrapper<double, int64_t>(double const *src_ptr,
                                                       int64_t *dst_ptr,
                                                       size_t volume);

template void backward_kernel_wrapper<int32_t, float>(int32_t const *src_ptr,
                                                      float *dst_ptr,
                                                      size_t volume);
template void backward_kernel_wrapper<int32_t, double>(int32_t const *src_ptr,
                                                       double *dst_ptr,
                                                       size_t volume);
template void backward_kernel_wrapper<int32_t, int32_t>(int32_t const *src_ptr,
                                                        int32_t *dst_ptr,
                                                        size_t volume);
template void backward_kernel_wrapper<int32_t, int64_t>(int32_t const *src_ptr,
                                                        int64_t *dst_ptr,
                                                        size_t volume);

template void backward_kernel_wrapper<int64_t, float>(int64_t const *src_ptr,
                                                      float *dst_ptr,
                                                      size_t volume);
template void backward_kernel_wrapper<int64_t, double>(int64_t const *src_ptr,
                                                       double *dst_ptr,
                                                       size_t volume);
template void backward_kernel_wrapper<int64_t, int32_t>(int64_t const *src_ptr,
                                                        int32_t *dst_ptr,
                                                        size_t volume);
template void backward_kernel_wrapper<int64_t, int64_t>(int64_t const *src_ptr,
                                                        int64_t *dst_ptr,
                                                        size_t volume);

namespace Internal {

template <typename IDT, typename ODT>
__global__ void cast_forward(IDT const *input, ODT *output, size_t volume) {
  CUDA_KERNEL_LOOP(i, volume) {
    output[i] = (ODT)input[i];
  }
}

template <typename IDT, typename ODT>
void forward_kernel(IDT const *input_ptr,
                    ODT *output_ptr,
                    size_t volume,
                    hipStream_t stream) {
  hipLaunchKernelGGL(HIP_KERNEL_NAME(cast_forward<IDT, ODT>),
                     GET_BLOCKS(volume),
                     CUDA_NUM_THREADS,
                     0,
                     stream,
                     input_ptr,
                     output_ptr,
                     volume);
}

template <typename IDT, typename ODT>
__global__ void
    cast_backward(IDT const *input, ODT *output, size_t volume, ODT beta) {
  CUDA_KERNEL_LOOP(i, volume) {
    output[i] = (ODT)input[i] + beta * output[i];
  }
}

template <typename IDT, typename ODT>
void backward_kernel(IDT const *src_ptr,
                     ODT *dst_ptr,
                     size_t volume,
                     hipStream_t stream) {
  hipLaunchKernelGGL(HIP_KERNEL_NAME(cast_backward<IDT, ODT>),
                     GET_BLOCKS(volume),
                     CUDA_NUM_THREADS,
                     0,
                     stream,
                     src_ptr,
                     dst_ptr,
                     volume,
                     (ODT)1.0f);
}

} // namespace Internal
} // namespace Cast
} // namespace Kernels
} // namespace FlexFlow
