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
#include "flexflow/ffconst_utils.h"
#include "flexflow/ops/argmax.h"
#include "flexflow/utils/cuda_helper.h"
#include <cub/cub.cuh>

namespace FlexFlow {

__global__ void
    half_2_float_array(half *ptr, float *ptr_f, int num_of_elements) {
  CUDA_KERNEL_LOOP(i, num_of_elements) {
    ptr_f[i] = __half2float(ptr[i]);
  }
}

__global__ void init_offset(int batch_size,
                            int vocab_size,
                            int total_eles,
                            int *d_offsets) {
  CUDA_KERNEL_LOOP(i, total_eles) {
    if (i % vocab_size == 0) {
      d_offsets[i / vocab_size] = i;
    }
  }
}

template <typename DT>
__global__ void copy_result(cub::KeyValuePair<int, DT> *d_out,
                            int *indices,
                            int batch_size) {
  CUDA_KERNEL_LOOP(i, batch_size) {
    indices[i] = d_out[i].key;
  }
}

/*static*/
template <typename DT>
void ArgMax::forward_kernel(ArgMaxMeta const *m,
                            DT *input_ptr,
                            int *indices_ptr,
                            DT *prob_ptr,
                            int *parent,
                            int const length,
                            int const batch_size,
                            cudaStream_t stream) {

  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
  DT alpha = 1.0f, beta = 0.0f;
  if (m->beam_search) {
    // set all parents id zero in arg top1 case.
    checkCUDA(cudaMemset(parent, 0, batch_size * sizeof(int)));
  }
  size_t temp_storage_bytes = m->temp_storage_bytes;
  // use cub
  checkCUDA(cub::DeviceSegmentedReduce::ArgMax(
      m->d_temp_storage,
      temp_storage_bytes,
      input_ptr,
      static_cast<cub::KeyValuePair<int, DT> *>(m->d_out),
      batch_size,
      m->d_offsets,
      m->d_offsets + 1,
      stream));

  // copy dout to incides
  int parallelism = batch_size;
  copy_result<<<GET_BLOCKS(parallelism),
                min(CUDA_NUM_THREADS, parallelism),
                0,
                stream>>>(static_cast<cub::KeyValuePair<int, DT> *>(m->d_out),
                          indices_ptr,
                          batch_size);
}

/*static*/
void ArgMax::forward_kernel_wrapper(ArgMaxMeta const *m,
                                    GenericTensorAccessorW const &input,
                                    GenericTensorAccessorW const &indices,
                                    GenericTensorAccessorW const &value,
                                    GenericTensorAccessorW const &parent,
                                    int batch_size) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  int length = input.domain.hi()[0] - input.domain.lo()[0] + 1;

  if (input.data_type == DT_HALF) {
    ArgMax::forward_kernel<half>(m,
                                 input.get_half_ptr(),
                                 indices.get_int32_ptr(),
                                 value.get_half_ptr(),
                                 m->beam_search ? parent.get_int32_ptr()
                                                : nullptr,
                                 length,
                                 batch_size,
                                 stream);
    if (m->beam_search) {
      half_2_float_array<<<GET_BLOCKS(batch_size),
                           CUDA_NUM_THREADS,
                           0,
                           stream>>>(
          value.get_half_ptr(), m->probs, batch_size);
    }

  } else if (input.data_type == DT_FLOAT) {
    ArgMax::forward_kernel<float>(m,
                                  input.get_float_ptr(),
                                  indices.get_int32_ptr(),
                                  value.get_float_ptr(),
                                  m->beam_search ? parent.get_int32_ptr()
                                                 : nullptr,
                                  length,
                                  batch_size,
                                  stream);
  } else {
    assert(false && "Unsupported data type");
  }

  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("[ArgMax] forward time = %.2lfms\n", elapsed);
  }
}

ArgMaxMeta::ArgMaxMeta(FFHandler handler,
                       Op const *op,
                       Legion::Domain const &input_domain,
                       Legion::Domain const &output_domain,
                       GenericTensorAccessorW input,
                       int batch_size,
                       int total_ele)
    : OpMeta(handler, op) {
  DataType data_type = op->data_type;

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  // init offset
  int parallelism = total_ele;
  cudaMalloc(&d_offsets, sizeof(int) * batch_size);
  init_offset<<<GET_BLOCKS(parallelism),
                min(CUDA_NUM_THREADS, parallelism),
                0,
                stream>>>(
      batch_size, total_ele / batch_size, total_ele, d_offsets);

  if (data_type == DT_FLOAT) {
    cudaMalloc(&d_out, sizeof(cub::KeyValuePair<int, float>) * batch_size);
    checkCUDA(cub::DeviceSegmentedReduce::ArgMax(
        d_temp_storage,
        temp_storage_bytes,
        input.get_float_ptr(),
        static_cast<cub::KeyValuePair<int, float> *>(d_out),
        batch_size,
        d_offsets,
        d_offsets + 1,
        stream));

  } else if (data_type == DT_HALF) {
    cudaMalloc(&d_out, sizeof(cub::KeyValuePair<int, half>) * batch_size);
    checkCUDA(cub::DeviceSegmentedReduce::ArgMax(
        d_temp_storage,
        temp_storage_bytes,
        input.get_half_ptr(),
        static_cast<cub::KeyValuePair<int, half> *>(d_out),
        batch_size,
        d_offsets,
        d_offsets + 1,
        stream));
  }

  cudaMalloc(&d_temp_storage, temp_storage_bytes);
}

}; // namespace FlexFlow