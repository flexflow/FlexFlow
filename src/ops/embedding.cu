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

#include "flexflow/ops/embedding.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {
// declare Legion names
using Legion::Context;
using Legion::Runtime;
using Legion::Domain;
using Legion::Task;
using Legion::Rect;
using Legion::PhysicalRegion;
using Legion::coord_t;


template<typename TI>
__global__
void embed_forward_no_aggr(const TI* input,
                           float* output,
                           const float* embed,
                           int out_dim,
                           int batch_size)
{
  CUDA_KERNEL_LOOP(i, batch_size * out_dim)
  {
    output[i] = 0;
    int idx = i / out_dim;
    int off = i % out_dim;
    TI wordIdx = input[idx];
    output[i] = embed[wordIdx * out_dim + off];
  }
}


template<typename TI>
__global__
void embed_forward_with_aggr(const TI* input,
                             float* output,
                             const float* embed,
                             int out_dim,
                             int in_dim,
                             int batch_size,
                             AggrMode aggr)
{
  CUDA_KERNEL_LOOP(i, batch_size * out_dim)
  {
    output[i] = 0;
    int idx = i / out_dim;
    int off = i % out_dim;
    for (int j = 0; j < in_dim; j++) {
      TI wordIdx = input[idx * in_dim + j];
      output[i] += embed[wordIdx * out_dim + off];
      if (aggr == AGGR_MODE_SUM) {
      } else {
        assert(aggr == AGGR_MODE_AVG);
        output[i] /= in_dim;
      }
    }
  }
}

template<typename TI>
__global__
void embed_backward_no_aggr(const TI* input,
                            const float* output,
                            float* embed,
                            int out_dim,
                            int batch_size) 
{
  CUDA_KERNEL_LOOP(i, batch_size * out_dim)
  {
    int idx = i / out_dim;
    int off = i % out_dim;
    TI wordIdx = input[idx];
    atomicAdd(embed + wordIdx * out_dim + off, output[i]);
  }
}

template<typename TI>
__global__
void embed_backward_with_aggr(const TI* input,
                              const float* output,
                              float* embed,
                              int out_dim,
                              int in_dim,
                              int batch_size,
                              AggrMode aggr)
{
  CUDA_KERNEL_LOOP(i, batch_size * out_dim)
  {
    int idx = i / out_dim;
    int off = i % out_dim;
    float gradient;
    if (aggr == AGGR_MODE_SUM) {
       gradient = output[i];
    } else {
      assert(aggr == AGGR_MODE_AVG);
      gradient = output[i] / in_dim;
    }
    for (int j = 0; j < in_dim; j++) {
      TI wordIdx = input[idx * in_dim + j];
      atomicAdd(embed + wordIdx * out_dim + off, gradient);
    }
  }
}

/*static*/
template<typename TI>
void Embedding::forward_kernel(const TI* input_ptr,
                               float *output_ptr,
                               float const *weight_ptr,
                               int in_dim,
                               int out_dim,
                               int batch_size,
                               AggrMode aggr,
                               int outputSize,
                               cudaStream_t stream)
{
  if (aggr == AGGR_MODE_NONE) {
    embed_forward_no_aggr<TI><<<GET_BLOCKS(outputSize), CUDA_NUM_THREADS, 0, stream>>>(
        input_ptr, output_ptr, weight_ptr,out_dim, batch_size);
  } else {
    embed_forward_with_aggr<TI><<<GET_BLOCKS(outputSize), CUDA_NUM_THREADS, 0, stream>>>(
        input_ptr, output_ptr, weight_ptr, out_dim, in_dim, batch_size, aggr);
  }
}

/*static*/
template<typename TI>
void Embedding::forward_kernel_wrapper(const EmbeddingMeta *m,
                                       const TI* input_ptr,
                                       float *output_ptr,
                                       float const *weight_ptr,
                                       int in_dim,
                                       int out_dim,
                                       int batch_size,
                                       AggrMode aggr,
                                       int outputSize)
{
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  Embedding::forward_kernel<TI>(input_ptr, output_ptr, weight_ptr,
                                in_dim, out_dim, batch_size,
                                m->aggr, outputSize, stream);

  if (m->profiling) {
    checkCUDA(cudaDeviceSynchronize());
    //print_tensor<TI>(input_ptr, input_domain.get_volume(), "[Embedding:forward:input]");
    //print_tensor<float>(kernel_ptr, kernel_domain.get_volume(), "[Embedding:forward:weight]");
    //print_tensor<float>(output_ptr, output_domain.get_volume(), "[Embedding:forward:output]");
  }
}

/*static*/
template<typename TI>
void Embedding::backward_kernel(const TI *input_ptr,
                                float const *output_ptr,
                                float *weight_grad_ptr,
                                int in_dim,
                                int out_dim,
                                int batch_size,
                                AggrMode aggr,
                                int outputSize,
                                cudaStream_t stream)
{
  if (aggr == AGGR_MODE_NONE) {
    embed_backward_no_aggr<TI><<<GET_BLOCKS(outputSize), CUDA_NUM_THREADS, 0, stream>>>(
        input_ptr, output_ptr, weight_grad_ptr, out_dim, batch_size);
  } else {
    embed_backward_with_aggr<TI><<<GET_BLOCKS(outputSize), CUDA_NUM_THREADS, 0, stream>>>(
        input_ptr, output_ptr, weight_grad_ptr, out_dim, in_dim, batch_size, aggr);
  }
}

/*static*/
template<typename TI>
void Embedding::backward_kernel_wrapper(const EmbeddingMeta *m,
                                        const TI *input_ptr,
                                        float const *output_ptr,
                                        float *weight_grad_ptr,
                                        int in_dim,
                                        int out_dim,
                                        int batch_size,
                                        AggrMode aggr,
                                        int outputSize)
{
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  Embedding::backward_kernel<TI>(input_ptr, output_ptr, weight_grad_ptr,
                                 in_dim, out_dim, batch_size,
                                 m->aggr, outputSize, stream);

  if (m->profiling) {
    checkCUDA(cudaDeviceSynchronize());
    //print_tensor<float>(output_grad_ptr, output_grad_domain.volume(), "[Embedding:backward:output_grad]");
    //print_tensor<float>(kernel_grad_ptr, kernel_grad_domain.get_volume(), "[Embedding:backward:weight_grad]");
    //print_tensor<TI>(input_ptr, input_domain.get_volume(), "[Embedding:backward:input]");
  }
}

__global__
void rand_generate_int64(int64_t* ptr, size_t size, int64_t p)
{
  CUDA_KERNEL_LOOP(i, size)
  {
    ptr[i] = i % p;
  }
}

void Embedding::rand_generate_int64_wrapper(int64_t* ptr, 
                                            size_t size, 
                                            int64_t p) const
{
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  // Randomly initialize the intput tensor to avoid out of index range issues
  rand_generate_int64<<<GET_BLOCKS(size), CUDA_NUM_THREADS, 0, stream>>>(
      ptr, size, p);
}

template void Embedding::forward_kernel_wrapper<int32_t>(const EmbeddingMeta *m, const int32_t* input_ptr, float *output_ptr, float const *weight_ptr, int in_dim, int out_dim, int batch_size, AggrMode aggr, int outputSize);
template void Embedding::forward_kernel_wrapper<int64_t>(const EmbeddingMeta *m, const int64_t* input_ptr, float *output_ptr, float const *weight_ptr, int in_dim, int out_dim, int batch_size, AggrMode aggr, int outputSize);

template void Embedding::backward_kernel_wrapper<int32_t>(const EmbeddingMeta *m, const int32_t *input_ptr, float const *output_ptr, float *weight_grad_ptr, int in_dim, int out_dim, int batch_size, AggrMode aggr, int outputSize);
template void Embedding::backward_kernel_wrapper<int64_t>(const EmbeddingMeta *m, const int64_t *input_ptr, float const *output_ptr, float *weight_grad_ptr, int in_dim, int out_dim, int batch_size, AggrMode aggr, int outputSize);

}; // namespace FlexFlow
