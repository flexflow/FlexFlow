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

#include "cub/cub.cuh"
#include "flexflow/ops/sampling.h"
#include "flexflow/utils/cuda_helper.h"
#include "flexflow/ffconst_utils.h"

namespace FlexFlow {

template <typename DT>
__global__ void mask_value_above_top_p(DT *input_ptr,
                                       DT *cumsum_ptr,
                                       float top_p,
                                       int total_eles) {
  CUDA_KERNEL_LOOP(i, total_eles) {
    if ((cumsum_ptr[i] - input_ptr[i]) > static_cast<DT>(top_p)) {
      input_ptr[i] = 0.0;
    }
  }
}

__global__ void init_idxs(int batch_size,
                          int vocab_size,
                          int total_eles,
                          int *idx,
                          int *begin_offset,
                          int *end_offset) {
  CUDA_KERNEL_LOOP(i, total_eles) {
    idx[i] = i % vocab_size;
    if (i % vocab_size == 0) {
      // printf("adfadf :%d\n", i);
      begin_offset[i / vocab_size] = i;
      end_offset[i / vocab_size] = i;
    }
  }
}

/*static*/
template <typename DT>
void Sampling::forward_kernel(SamplingMeta const *m,
                              DT *input_ptr,
                              int *indices_ptr,
                              float top_p,
                              int length,
                              int batch_size,
                              cudaStream_t stream) {
  // 1. sort
  // 2. cumsum
  // how to do it in parallel?
  // init
  print_tensor<float>((float *)input_ptr+ 32000, 32, "inputttt");
  std::cout<< "meta " << length << ", " << batch_size << "\n";
  int parallelism = length * batch_size;
  init_idxs<<<GET_BLOCKS(parallelism),
              min(CUDA_NUM_THREADS, parallelism),
              0,
              stream>>>(batch_size,
                        length,
                        length * batch_size,
                        m->idx,
                        m->begin_offset,
                        m->end_offset);

  checkCUDA(cudaDeviceSynchronize());                         
  // print_tensor<int>(m->begin_offset, 64, "ofsset");
  // print_tensor<int>(m->end_offset, 64, "ofsset");  

  std::cout<<"-------------------------sampling kernel _--------------------" << "\n";                                 
  // sort
  size_t temp_storage_bytes = 0;
  void *d_temp_storage = nullptr;
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      m->d_temp_storage,
      temp_storage_bytes,
      input_ptr,
      static_cast<DT *>(m->sorted_logits),
      m->idx,
      m->sorted_idx,
      length * batch_size,
      batch_size,
      m->begin_offset,
      m->end_offset + 1,
      0,              // begin_bit
      sizeof(DT) * 8, // end_bit = sizeof(KeyT) * 8
      stream);

  
  checkCUDA(cudaDeviceSynchronize()); 
  cudaMalloc(&d_temp_storage, temp_storage_bytes);  
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      d_temp_storage,
      temp_storage_bytes,
      input_ptr,
      static_cast<DT *>(m->sorted_logits),
      m->idx,
      m->sorted_idx,
      length * batch_size,
      batch_size,
      m->begin_offset,
      m->end_offset + 1,
      0,              // begin_bit
      sizeof(DT) * 8, // end_bit = sizeof(KeyT) * 8
      stream);
  print_tensor<float>((float *)m->sorted_logits + 32000, 32, "after sort");    
  print_tensor<int>(m->sorted_idx+ 32000, 32, "after sort");
  // print_tensor<int>((int *)indices_ptr, 15, "sdsdasd");
  assert(false);
}

/*static*/
void Sampling::forward_kernel_wrapper(SamplingMeta const *m,
                                      GenericTensorAccessorW const &input,
                                      GenericTensorAccessorW const &indices) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  int length = input.domain.hi()[0] - input.domain.lo()[0] + 1;
  int batch_size = input.domain.get_volume() / length;

  if (input.data_type == DT_HALF) {
    Sampling::forward_kernel<half>(m,
                                   input.get_half_ptr(),
                                   indices.get_int32_ptr(),
                                   m->top_p,
                                   length,
                                   batch_size,
                                   stream);
  } else if (input.data_type == DT_FLOAT) {
    Sampling::forward_kernel<float>(m,
                                    input.get_float_ptr(),
                                    indices.get_int32_ptr(),
                                    m->top_p,
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
    printf("[Sampling] forward time = %.2lfms\n", elapsed);
  }
}

SamplingMeta::SamplingMeta(FFHandler handler,
                           Op const *op,
                           int batch_size,
                           int total_ele)
    : OpMeta(handler, op) {
  DataType data_type = op->data_type;
  checkCUDA(cudaMalloc(&begin_offset, (batch_size + 1) * sizeof(int)));
  checkCUDA(cudaMalloc(&end_offset, (batch_size + 1) * sizeof(int)));
  checkCUDA(cudaMalloc(&idx, total_ele * sizeof(int)));

  checkCUDA(cudaMalloc(&sorted_idx, total_ele * sizeof(int)));
  checkCUDA(cudaMalloc(&sorted_logits, total_ele * data_type_size(data_type)));
  
}

}; // namespace FlexFlow