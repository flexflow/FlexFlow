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

#include "flexflow/ops/kernels/embedding_kernels.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {
// declare Legion names
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::PhysicalRegion;
using Legion::Rect;
using Legion::Runtime;
using Legion::Task;

namespace Kernels {
namespace Embedding {

/*static*/
void forward_kernel_wrapper(EmbeddingMeta const *m,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorW const &output,
                            GenericTensorAccessorR const &weight,
                            int in_dim,
                            int out_dim,
                            int batch_size) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  if (input.data_type == DT_INT32) {
    if (weight.data_type == DT_HALF) {
      Internal::forward_kernel(input.get_int32_ptr(),
                               output.get_half_ptr(),
                               weight.get_half_ptr(),
                               in_dim,
                               out_dim,
                               batch_size,
                               m->aggr,
                               output.domain.get_volume(),
                               stream);
    } else if (weight.data_type == DT_FLOAT) {
      Internal::forward_kernel(input.get_int32_ptr(),
                               output.get_float_ptr(),
                               weight.get_float_ptr(),
                               in_dim,
                               out_dim,
                               batch_size,
                               m->aggr,
                               output.domain.get_volume(),
                               stream);
    } else if (weight.data_type == DT_DOUBLE) {
      Internal::forward_kernel(input.get_int32_ptr(),
                               output.get_double_ptr(),
                               weight.get_double_ptr(),
                               in_dim,
                               out_dim,
                               batch_size,
                               m->aggr,
                               output.domain.get_volume(),
                               stream);
    } else {
      assert(false && "Unsupported DataType in Embedding");
    }
  } else if (input.data_type == DT_INT64) {
    if (weight.data_type == DT_HALF) {
      Internal::forward_kernel(input.get_int64_ptr(),
                               output.get_half_ptr(),
                               weight.get_half_ptr(),
                               in_dim,
                               out_dim,
                               batch_size,
                               m->aggr,
                               output.domain.get_volume(),
                               stream);
    } else if (weight.data_type == DT_FLOAT) {
      Internal::forward_kernel(input.get_int64_ptr(),
                               output.get_float_ptr(),
                               weight.get_float_ptr(),
                               in_dim,
                               out_dim,
                               batch_size,
                               m->aggr,
                               output.domain.get_volume(),
                               stream);
    } else if (weight.data_type == DT_DOUBLE) {
      Internal::forward_kernel(input.get_int64_ptr(),
                               output.get_double_ptr(),
                               weight.get_double_ptr(),
                               in_dim,
                               out_dim,
                               batch_size,
                               m->aggr,
                               output.domain.get_volume(),
                               stream);
    } else {
      assert(false && "Unsupported DataType in Embedding");
    }
  } else {
    assert(false && "Unsupported DataType in Embedding");
  }
  if (m->profiling) {
    checkCUDA(cudaDeviceSynchronize());
    // print_tensor<TI>(input_ptr, input_domain.get_volume(),
    // "[Embedding:forward:input]"); print_tensor<float>(kernel_ptr,
    // kernel_domain.get_volume(), "[Embedding:forward:weight]");
    // print_tensor<float>(output_ptr, output_domain.get_volume(),
    // "[Embedding:forward:output]");
  }
}

/*static*/
void backward_kernel_wrapper(EmbeddingMeta const *m,
                             GenericTensorAccessorR const &input,
                             GenericTensorAccessorR const &output,
                             GenericTensorAccessorW const &weight_grad,
                             int in_dim,
                             int out_dim,
                             int batch_size) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  if (m->input_type[0] == DT_INT32) {
    if (m->output_type[0] == DT_HALF) {
      Internal::backward_kernel(input.get_int32_ptr(),
                                output.get_half_ptr(),
                                weight_grad.get_half_ptr(),
                                in_dim,
                                out_dim,
                                batch_size,
                                m->aggr,
                                output.domain.get_volume(),
                                stream);
    } else if (m->output_type[0] == DT_FLOAT) {
      Internal::backward_kernel(input.get_int32_ptr(),
                                output.get_float_ptr(),
                                weight_grad.get_float_ptr(),
                                in_dim,
                                out_dim,
                                batch_size,
                                m->aggr,
                                output.domain.get_volume(),
                                stream);
    } else if (m->output_type[0] == DT_DOUBLE) {
      Internal::backward_kernel(input.get_int32_ptr(),
                                output.get_double_ptr(),
                                weight_grad.get_double_ptr(),
                                in_dim,
                                out_dim,
                                batch_size,
                                m->aggr,
                                output.domain.get_volume(),
                                stream);
    } else {
      assert(false && "Unsupported DataType in Embedding");
    }
  } else if (m->input_type[0] == DT_INT64) {
    if (m->output_type[0] == DT_HALF) {
      Internal::backward_kernel(input.get_int64_ptr(),
                                output.get_half_ptr(),
                                weight_grad.get_half_ptr(),
                                in_dim,
                                out_dim,
                                batch_size,
                                m->aggr,
                                output.domain.get_volume(),
                                stream);
    } else if (m->output_type[0] == DT_FLOAT) {
      Internal::backward_kernel(input.get_int64_ptr(),
                                output.get_float_ptr(),
                                weight_grad.get_float_ptr(),
                                in_dim,
                                out_dim,
                                batch_size,
                                m->aggr,
                                output.domain.get_volume(),
                                stream);
    } else if (m->output_type[0] == DT_DOUBLE) {
      Internal::backward_kernel(input.get_int64_ptr(),
                                output.get_double_ptr(),
                                weight_grad.get_double_ptr(),
                                in_dim,
                                out_dim,
                                batch_size,
                                m->aggr,
                                output.domain.get_volume(),
                                stream);
    } else {
      assert(false && "Unsupported DataType in Embedding");
    }
  }

  if (m->profiling) {
    checkCUDA(cudaDeviceSynchronize());
    // print_tensor<float>(output_grad_ptr, output_grad_domain.volume(),
    // "[Embedding:backward:output_grad]"); print_tensor<float>(kernel_grad_ptr,
    // kernel_grad_domain.get_volume(), "[Embedding:backward:weight_grad]");
    // print_tensor<TI>(input_ptr, input_domain.get_volume(),
    // "[Embedding:backward:input]");
  }
}

void rand_generate_int64_wrapper(int64_t *ptr, size_t size, int64_t p) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  // Randomly initialize the intput tensor to avoid out of index range issues
  Internal::
      rand_generate_int<<<GET_BLOCKS(size), CUDA_NUM_THREADS, 0, stream>>>(
          ptr, size, p);
}

void rand_generate_int32_wrapper(int32_t *ptr, size_t size, int32_t p) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  // Randomly initialize the intput tensor to avoid out of index range issues
  Internal::
      rand_generate_int<<<GET_BLOCKS(size), CUDA_NUM_THREADS, 0, stream>>>(
          ptr, size, p);
}

namespace Internal {

template <typename TI, typename TD>
__global__ void embed_forward_no_aggr(
    TI const *input, TD *output, TD const *embed, int out_dim, int batch_size) {
  CUDA_KERNEL_LOOP(i, batch_size * out_dim) {
    output[i] = 0;
    int idx = i / out_dim;
    int off = i % out_dim;
    TI wordIdx = input[idx];
    output[i] = embed[wordIdx * out_dim + off];
  }
}

template <typename TI, typename TD>
__global__ void embed_forward_with_aggr(TI const *input,
                                        TD *output,
                                        TD const *embed,
                                        int out_dim,
                                        int in_dim,
                                        int batch_size,
                                        AggrMode aggr) {
  TD scale = 1.0f / in_dim;
  CUDA_KERNEL_LOOP(i, batch_size * out_dim) {
    output[i] = 0;
    int idx = i / out_dim;
    int off = i % out_dim;
    for (int j = 0; j < in_dim; j++) {
      TI wordIdx = input[idx * in_dim + j];
      output[i] = output[i] + embed[wordIdx * out_dim + off];
      if (aggr == AGGR_MODE_SUM) {
      } else {
        assert(aggr == AGGR_MODE_AVG);
        output[i] = output[i] * scale;
      }
    }
  }
}

template <typename TI, typename TD>
__global__ void embed_backward_no_aggr(
    TI const *input, TD const *output, TD *embed, int out_dim, int batch_size) {
  CUDA_KERNEL_LOOP(i, batch_size * out_dim) {
    int idx = i / out_dim;
    int off = i % out_dim;
    TI wordIdx = input[idx];
    atomicAdd(embed + wordIdx * out_dim + off, output[i]);
  }
}

// Specialization for half type

template <>
__global__ void embed_backward_no_aggr<int, half>(int const *input,
                                                  half const *output,
                                                  half *embed,
                                                  int out_dim,
                                                  int batch_size) {
  CUDA_KERNEL_LOOP(i, batch_size * out_dim) {
    int idx = i / out_dim;
    int off = i % out_dim;
    int wordIdx = input[idx];
#if __CUDA_ARCH__ >= 700
    atomicAdd(embed + wordIdx * out_dim + off, output[i]);
#else
    assert(false);
    // TODO: this implementation may result in race condition
    // so we use an assertion failure to warn users
    embed[wordIdx * out_dim + off] += output[i];
#endif
  }
}

template <>
__global__ void embed_backward_no_aggr<int64_t, half>(int64_t const *input,
                                                      half const *output,
                                                      half *embed,
                                                      int out_dim,
                                                      int batch_size) {
  CUDA_KERNEL_LOOP(i, batch_size * out_dim) {
    int idx = i / out_dim;
    int off = i % out_dim;
    int64_t wordIdx = input[idx];
#if __CUDA_ARCH__ >= 700
    atomicAdd(embed + wordIdx * out_dim + off, output[i]);
#else
    assert(false);
    // TODO: this implementation may result in race condition
    // so we use an assertion failure to warn users
    embed[wordIdx * out_dim + off] += output[i];
#endif
  }
}

template <typename TI, typename TD>
__global__ void embed_backward_with_aggr(TI const *input,
                                         TD const *output,
                                         TD *embed,
                                         int out_dim,
                                         int in_dim,
                                         int batch_size,
                                         AggrMode aggr) {
  TD scale = 1.0f / in_dim;
  CUDA_KERNEL_LOOP(i, batch_size * out_dim) {
    int idx = i / out_dim;
    int off = i % out_dim;
    TD gradient;
    if (aggr == AGGR_MODE_SUM) {
      gradient = output[i];
    } else {
      assert(aggr == AGGR_MODE_AVG);
      gradient = output[i] * scale;
    }
    for (int j = 0; j < in_dim; j++) {
      TI wordIdx = input[idx * in_dim + j];
      atomicAdd(embed + wordIdx * out_dim + off, gradient);
    }
  }
}

// Specialization for half type

template <>
__global__ void embed_backward_with_aggr<int, half>(int const *input,
                                                    half const *output,
                                                    half *embed,
                                                    int out_dim,
                                                    int in_dim,
                                                    int batch_size,
                                                    AggrMode aggr) {
  half scale = 1.0f / in_dim;
  CUDA_KERNEL_LOOP(i, batch_size * out_dim) {
    int idx = i / out_dim;
    int off = i % out_dim;
    half gradient;
    if (aggr == AGGR_MODE_SUM) {
      gradient = output[i];
    } else {
      assert(aggr == AGGR_MODE_AVG);
      gradient = output[i] * scale;
    }
    for (int j = 0; j < in_dim; j++) {
      int wordIdx = input[idx * in_dim + j];
#if __CUDA_ARCH__ >= 700
      atomicAdd(embed + wordIdx * out_dim + off, gradient);
#else
      assert(false);
      // TODO: this implementation may result in race condition
      // so we use an assertion failure to warn users
      embed[wordIdx * out_dim + off] += gradient;
#endif
    }
  }
}

template <>
__global__ void embed_backward_with_aggr<int64_t, half>(int64_t const *input,
                                                        half const *output,
                                                        half *embed,
                                                        int out_dim,
                                                        int in_dim,
                                                        int batch_size,
                                                        AggrMode aggr) {
  half scale = 1.0f / in_dim;
  CUDA_KERNEL_LOOP(i, batch_size * out_dim) {
    int idx = i / out_dim;
    int off = i % out_dim;
    half gradient;
    if (aggr == AGGR_MODE_SUM) {
      gradient = output[i];
    } else {
      assert(aggr == AGGR_MODE_AVG);
      gradient = output[i] * scale;
    }
    for (int j = 0; j < in_dim; j++) {
      int64_t wordIdx = input[idx * in_dim + j];
#if __CUDA_ARCH__ >= 700
      atomicAdd(embed + wordIdx * out_dim + off, gradient);
#else
      assert(false);
      // TODO: this implementation may result in race condition
      // so we use an assertion failure to warn users
      embed[wordIdx * out_dim + off] += gradient;
#endif
    }
  }
}

/*static*/
template <typename TI, typename TD>
void forward_kernel(TI const *input_ptr,
                    TD *output_ptr,
                    TD const *weight_ptr,
                    int in_dim,
                    int out_dim,
                    int batch_size,
                    AggrMode aggr,
                    int outputSize,
                    cudaStream_t stream) {
  assert(input_ptr != nullptr);
  assert(output_ptr != nullptr);
  assert(weight_ptr != nullptr);

  if (aggr == AGGR_MODE_NONE) {
    embed_forward_no_aggr<TI, TD>
        <<<GET_BLOCKS(outputSize), CUDA_NUM_THREADS, 0, stream>>>(
            input_ptr, output_ptr, weight_ptr, out_dim, batch_size);
  } else {
    assert(aggr == AGGR_MODE_AVG || aggr == AGGR_MODE_SUM);
    embed_forward_with_aggr<TI, TD>
        <<<GET_BLOCKS(outputSize), CUDA_NUM_THREADS, 0, stream>>>(input_ptr,
                                                                  output_ptr,
                                                                  weight_ptr,
                                                                  out_dim,
                                                                  in_dim,
                                                                  batch_size,
                                                                  aggr);
  }
}

/*static*/
template <typename TI, typename TD>
void backward_kernel(TI const *input_ptr,
                     TD const *output_ptr,
                     TD *weight_grad_ptr,
                     int in_dim,
                     int out_dim,
                     int batch_size,
                     AggrMode aggr,
                     int outputSize,
                     cudaStream_t stream) {
  assert(input_ptr != nullptr);
  assert(output_ptr != nullptr);
  assert(weight_grad_ptr != nullptr);
  if (aggr == AGGR_MODE_NONE) {
    embed_backward_no_aggr<TI, TD>
        <<<GET_BLOCKS(outputSize), CUDA_NUM_THREADS, 0, stream>>>(
            input_ptr, output_ptr, weight_grad_ptr, out_dim, batch_size);
  } else {
    embed_backward_with_aggr<TI, TD>
        <<<GET_BLOCKS(outputSize), CUDA_NUM_THREADS, 0, stream>>>(
            input_ptr,
            output_ptr,
            weight_grad_ptr,
            out_dim,
            in_dim,
            batch_size,
            aggr);
  }
}

template <typename TD>
__global__ void rand_generate_int(TD *ptr, size_t size, TD p) {
  CUDA_KERNEL_LOOP(i, size) {
    ptr[i] = i % p;
  }
}

#ifdef DEADCODE
template void forward_kernel_wrapper<int32_t>(EmbeddingMeta const *m,
                                              int32_t const *input_ptr,
                                              float *output_ptr,
                                              float const *weight_ptr,
                                              int in_dim,
                                              int out_dim,
                                              int batch_size,
                                              AggrMode aggr,
                                              int outputSize);
template void forward_kernel_wrapper<int64_t>(EmbeddingMeta const *m,
                                              int64_t const *input_ptr,
                                              float *output_ptr,
                                              float const *weight_ptr,
                                              int in_dim,
                                              int out_dim,
                                              int batch_size,
                                              AggrMode aggr,
                                              int outputSize);

template void backward_kernel_wrapper<int32_t>(EmbeddingMeta const *m,
                                               int32_t const *input_ptr,
                                               float const *output_ptr,
                                               float *weight_grad_ptr,
                                               int in_dim,
                                               int out_dim,
                                               int batch_size,
                                               AggrMode aggr,
                                               int outputSize);
template void backward_kernel_wrapper<int64_t>(EmbeddingMeta const *m,
                                               int64_t const *input_ptr,
                                               float const *output_ptr,
                                               float *weight_grad_ptr,
                                               int in_dim,
                                               int out_dim,
                                               int batch_size,
                                               AggrMode aggr,
                                               int outputSize);
#endif
} // namespace Internal
} // namespace Embedding
} // namespace Kernels
}; // namespace FlexFlow
