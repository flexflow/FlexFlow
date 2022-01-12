/* Copyright 2021 CMU
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

#include "flexflow/ops/cast.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

template<typename IDT, typename ODT>
__global__
void cast_forward(const IDT* input,
                  ODT* output,
                  size_t volume) 
{
  CUDA_KERNEL_LOOP(i, volume) {
    output[i] = (ODT) input[i];
  }
}

/*static*/
template<typename IDT, typename ODT>
void Cast::forward_kernel(const IDT* input_ptr,
                          ODT* output_ptr,
                          size_t volume,
                          cudaStream_t stream) 
{
  cast_forward<IDT, ODT><<<GET_BLOCKS(volume), CUDA_NUM_THREADS, 0, stream>>>(
      input_ptr, output_ptr, volume);
}

/*static*/
template<typename IDT, typename ODT>
void Cast::forward_kernel_wrapper(const IDT* input_ptr,
                                  ODT* output_ptr,
                                  size_t volume) 
{
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  Cast::forward_kernel<IDT, ODT>(input_ptr, output_ptr, volume, stream);
}

template<typename IDT, typename ODT>
__global__
void cast_backward(const IDT* input,
                   ODT* output,
                   size_t volume,
                   ODT beta) 
{
  CUDA_KERNEL_LOOP(i, volume) {
    output[i] = (ODT) input[i] + beta * output[i];
  }
}

/*static*/
template<typename IDT, typename ODT>
void Cast::backward_kernel(const IDT* src_ptr,
                           ODT* dst_ptr,
                           size_t volume,
                           cudaStream_t stream) 
{
  cast_backward<IDT, ODT><<<GET_BLOCKS(volume), CUDA_NUM_THREADS, 0, stream>>>(
      src_ptr, dst_ptr, volume, (ODT)1.0f);
}

/*static*/
template<typename IDT, typename ODT>
void Cast::backward_kernel_wrapper(const IDT* src_ptr,
                                   ODT* dst_ptr,
                                   size_t volume) 
{
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  Cast::backward_kernel<IDT, ODT>(src_ptr, dst_ptr, volume, stream);
}

CastMeta::CastMeta(FFHandler handle)
: OpMeta(handle) {}

template void Cast::forward_kernel_wrapper<float, float>(const float* input_ptr, float* output_ptr, size_t volume);
template void Cast::forward_kernel_wrapper<float, double>(const float* input_ptr, double* output_ptr, size_t volume);
template void Cast::forward_kernel_wrapper<float, int32_t>(const float* input_ptr, int32_t* output_ptr, size_t volume);
template void Cast::forward_kernel_wrapper<float, int64_t>(const float* input_ptr, int64_t* output_ptr, size_t volume);

template void Cast::forward_kernel_wrapper<double, float>(const double* input_ptr, float* output_ptr, size_t volume);
template void Cast::forward_kernel_wrapper<double, double>(const double* input_ptr, double* output_ptr, size_t volume);
template void Cast::forward_kernel_wrapper<double, int32_t>(const double* input_ptr, int32_t* output_ptr, size_t volume);
template void Cast::forward_kernel_wrapper<double, int64_t>(const double* input_ptr, int64_t* output_ptr, size_t volume);

template void Cast::forward_kernel_wrapper<int32_t, float>(const int32_t* input_ptr, float* output_ptr, size_t volume);
template void Cast::forward_kernel_wrapper<int32_t, double>(const int32_t* input_ptr, double* output_ptr, size_t volume);
template void Cast::forward_kernel_wrapper<int32_t, int32_t>(const int32_t* input_ptr, int32_t* output_ptr, size_t volume);
template void Cast::forward_kernel_wrapper<int32_t, int64_t>(const int32_t* input_ptr, int64_t* output_ptr, size_t volume);

template void Cast::forward_kernel_wrapper<int64_t, float>(const int64_t* input_ptr, float* output_ptr, size_t volume);
template void Cast::forward_kernel_wrapper<int64_t, double>(const int64_t* input_ptr, double* output_ptr, size_t volume);
template void Cast::forward_kernel_wrapper<int64_t, int32_t>(const int64_t* input_ptr, int32_t* output_ptr, size_t volume);
template void Cast::forward_kernel_wrapper<int64_t, int64_t>(const int64_t* input_ptr, int64_t* output_ptr, size_t volume);

template void Cast::backward_kernel_wrapper<float, float>(const float* src_ptr, float* dst_ptr, size_t volume);
template void Cast::backward_kernel_wrapper<float, double>(const float* src_ptr, double* dst_ptr, size_t volume);
template void Cast::backward_kernel_wrapper<float, int32_t>(const float* src_ptr, int32_t* dst_ptr, size_t volume);
template void Cast::backward_kernel_wrapper<float, int64_t>(const float* src_ptr, int64_t* dst_ptr, size_t volume);

template void Cast::backward_kernel_wrapper<double, float>(const double* src_ptr, float* dst_ptr, size_t volume);
template void Cast::backward_kernel_wrapper<double, double>(const double* src_ptr, double* dst_ptr, size_t volume);
template void Cast::backward_kernel_wrapper<double, int32_t>(const double* src_ptr, int32_t* dst_ptr, size_t volume);
template void Cast::backward_kernel_wrapper<double, int64_t>(const double* src_ptr, int64_t* dst_ptr, size_t volume);

template void Cast::backward_kernel_wrapper<int32_t, float>(const int32_t* src_ptr, float* dst_ptr, size_t volume);
template void Cast::backward_kernel_wrapper<int32_t, double>(const int32_t* src_ptr, double* dst_ptr, size_t volume);
template void Cast::backward_kernel_wrapper<int32_t, int32_t>(const int32_t* src_ptr, int32_t* dst_ptr, size_t volume);
template void Cast::backward_kernel_wrapper<int32_t, int64_t>(const int32_t* src_ptr, int64_t* dst_ptr, size_t volume);

template void Cast::backward_kernel_wrapper<int64_t, float>(const int64_t* src_ptr, float* dst_ptr, size_t volume);
template void Cast::backward_kernel_wrapper<int64_t, double>(const int64_t* src_ptr, double* dst_ptr, size_t volume);
template void Cast::backward_kernel_wrapper<int64_t, int32_t>(const int64_t* src_ptr, int32_t* dst_ptr, size_t volume);
template void Cast::backward_kernel_wrapper<int64_t, int64_t>(const int64_t* src_ptr, int64_t* dst_ptr, size_t volume);

}; //namespace FlexFlow
