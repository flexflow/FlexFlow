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
#include "flexflow/ffconst_utils.h"
#include "flexflow/ops/sampling.h"
#include "flexflow/utils/cuda_helper.h"
#include <curand.h>
#include <curand_kernel.h>

namespace FlexFlow {

struct BlockPrefixCallbackOp {
  // Running prefix
  float running_total;
  // Constructor
  __device__ BlockPrefixCallbackOp(float running_total)
      : running_total(running_total) {}
  // Callback operator to be entered by the first warp of threads in the block.
  // Thread-0 is responsible for returning a value for seeding the block-wide
  // scan.
  __device__ float operator()(float block_aggregate) {
    float old_prefix = running_total;
    running_total += block_aggregate;
    return old_prefix;
  }
};

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

__global__ void
    init_random_kernel(curandState *state, int batch_size, long rand) {
  CUDA_KERNEL_LOOP(i, batch_size) {
    curand_init(rand, i, 0, &state[i]);
  }
}

// multinominal and gather
template <typename DT, int BLOCK_SIZE>
__global__ void sampling_topp_kernel(int batch_size,
                                     int const vocab_size,
                                     curandState *state,
                                     DT *sorted_logits,
                                     int *sorted_idx,
                                     int *indices_ptr,
                                     float topp) {
  int const vocab_id = threadIdx.x;
  int const batch_idx = blockIdx.x;
  __shared__ float random_n;
  __shared__ float renormalized_sum;
  __shared__ long long result_idx;

  // random num
  if (threadIdx.x == 0) {
    // number must < topp
    random_n = curand_uniform(state + batch_idx) * topp;
    printf("batch idx: %d, %f\n", batch_idx, random_n);
  }

  __syncthreads();

  // cumsum;
  typedef cub::BlockScan<float, BLOCK_SIZE> BlockScan;
  typedef cub::BlockScan<float, BLOCK_SIZE> BlockScanMultiNominal;
  typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
  __shared__ typename BlockScan::TempStorage temp_storage;
  __shared__ typename BlockReduce::TempStorage reduce_temp_storage;
  __shared__ typename BlockScan::TempStorage multinominal_temp_storage;

  int offset = batch_idx * vocab_size;
  float prefix_sum = 0.0f;
  int end = ((vocab_size + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
  BlockPrefixCallbackOp prefix_op(0);
  float sum;
  result_idx = vocab_size;

  for (long long j = threadIdx.x; j < vocab_size; j += blockDim.x) {
    float logit = (float)sorted_logits[offset + j];
    // float logit = (j < vocab_size) ? (float)sorted_logits[offset + j] : 0.f;
    BlockScan(temp_storage).InclusiveSum(logit, prefix_sum, prefix_op);

    prefix_sum /= topp;

    if (prefix_sum >= random_n) {
      atomicMin(&result_idx, j);
    }

    // if (blockIdx.x == 0 && j == 276){
    //   printf("batch idx afterward aaaaaa: %f, %.10f, %.10f, %.10f, %.10f\n",
    //   topp, prefix_sum, logit,  (float)sorted_logits[offset + j], random_n);
    // }
    // if (blockIdx.x == 1 && j == 39){
    //   printf("batch idx afterward aaaaaa11111: %f, %.10f, %.10f, %.10f,
    //   %.10f\n", topp, prefix_sum, logit,  (float)sorted_logits[offset + j],
    //   random_n);
    // }

    // // mask
    // sorted_logits[offset + j] =
    //     (prefix_sum - (float)sorted_logits[offset + j] > topp)
    //         ? (DT)0
    //         : sorted_logits[offset + j];

    // //get sum and divide
    // sum += (float)sorted_logits[offset + j];
    // __syncthreads();
    // if (blockIdx.x == 0 && j > 31990) {
    //   printf(
    //       "batch idx afterward after:%d,  %.20f, %.20f\n", j, prefix_sum,
    //       logit);
    // }
    // if (blockIdx.x == 0 && j > 1022 && j < 1028) {
    //   printf(
    //       "batch idx afterward before:%d,  %,20f, %.20f\n", j, prefix_sum,
    //       logit);
    // }
  }

  indices_ptr[batch_idx] = sorted_idx[offset + result_idx];
  // if meet latency issue, this part can also be removed because the sum is
  // very close to topp.
  //  float temp_sum = BlockReduce(reduce_temp_storage).Sum(sum);
  //  __syncthreads();
  //  if(threadIdx.x == 0){
  //    renormalized_sum = temp_sum;
  //  }
  //  __syncthreads();

  // renormalized and multinominal
  //  result_idx = vocab_size;
  //  BlockPrefixCallbackOp prefix_op_2(0);
  //  prefix_sum = 0.0f;
  //  for (long long j = threadIdx.x; j < vocab_size; j += blockDim.x) {
  //    float logit = (float)sorted_logits[offset + j] / topp;
  //    BlockScanMultiNominal(multinominal_temp_storage).InclusiveSum(logit,
  //    prefix_sum, prefix_op_2);

  //   if(prefix_sum >= random_n){
  //       atomicMin(&result_idx, j);
  //   }

  //   if (blockIdx.x == 0 && j == 1023){
  //     printf("batch idx afterward aaaaaa: %f, %.10f, %.10f, %.10f, %.10f\n",
  //     topp, prefix_sum, logit,  (float)sorted_logits[offset + j], random_n);
  //   }
  //   if (blockIdx.x == 1 && j == 39){
  //     printf("batch idx afterward aaaaaa11111: %f, %.10f, %.10f, %.10f,
  //     %.10f\n", topp, prefix_sum, logit,  (float)sorted_logits[offset + j],
  //     random_n);
  //   }
  // }
  // indices_ptr[batch_idx] = (int)result_idx;

  // __syncthreads();

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("batch idx afterward aaaaaa: %d\n", result_idx);
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("batch idx afterward aaaaaa0000: %d\n", result_idx);
  }
  if (blockIdx.x == 1 && threadIdx.x == 1) {
    printf("batch idx afterward aaaaaa11111: %d\n", result_idx);
  }

  // if (threadIdx.x == 1) {
  //   printf("batch idx afterward: %d, %f, %f, %d\n", batch_idx, prefix_sum,
  //   (float)sorted_logits[offset], offset);
  // }

  // mask, div

  // select
}

/*static*/
template <typename DT>
void Sampling::forward_kernel(SamplingMeta const *m,
                              DT *input_ptr,
                              int *indices_ptr,
                              float const top_p,
                              int const length,
                              int const batch_size,
                              cudaStream_t stream) {
  // 1. sort
  // 2. cumsum
  // how to do it in parallel?
  // init
  print_tensor<float>((float *)input_ptr + 32000, 32, "inputttt");
  std::cout << "meta " << length << ", " << batch_size << "\n";
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

  std::cout << "-------------------------sampling kernel _--------------------"
            << "\n";
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
  // sort
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
  print_tensor<float>((float *)m->sorted_logits, 32, "after sort 0");
  // print_tensor<float>((float *)m->sorted_logits + 31990, 32, "after sort 1");
  // print_tensor<int>((int *)indices_ptr, 15, "sdsdasd");

  // random

  parallelism = batch_size;
  init_random_kernel<<<GET_BLOCKS(parallelism),
                       min(CUDA_NUM_THREADS, parallelism),
                       0,
                       stream>>>(m->state, batch_size, rand());
  sampling_topp_kernel<DT, 1024>
      <<<batch_size, 1024, 0, stream>>>(batch_size,
                                        length,
                                        m->state,
                                        static_cast<DT *>(m->sorted_logits),
                                        m->sorted_idx,
                                        indices_ptr,
                                        0.95f);

  checkCUDA(cudaDeviceSynchronize());
  // print_tensor<float>((float *)m->sorted_logits + 32000, 32, "after sort");
  // topk / topp mask some value and renormalize

  // sampling
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
  cudaMalloc(&state, sizeof(curandState) * batch_size);
}

}; // namespace FlexFlow