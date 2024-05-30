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

constexpr int SamplingNumThreads = 1024;
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

__global__ void init_idxs(int batch_size,
                          int vocab_size,
                          int total_eles,
                          int *idx,
                          int *begin_offset,
                          int *end_offset) {
  CUDA_KERNEL_LOOP(i, total_eles) {
    idx[i] = i % vocab_size;
    if (i % vocab_size == 0) {
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
  // int const vocab_id = threadIdx.x;
  int const batch_idx = blockIdx.x;
  __shared__ float random_n;
  __shared__ long long result_idx;

  // random num
  if (threadIdx.x == 0) {
    // number must < topp
    random_n = curand_uniform(state + batch_idx) * topp;
    // printf("batch idx: %d, random num%f\n", batch_idx, random_n);
  }

  __syncthreads();

  // cumsum;
  typedef cub::BlockScan<float, BLOCK_SIZE> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  int offset = batch_idx * vocab_size;
  float prefix_sum = 0.0f;
  BlockPrefixCallbackOp prefix_op(0);
  result_idx = vocab_size - 1;

  for (long long j = threadIdx.x; j < vocab_size; j += blockDim.x) {
    float logit = (float)(sorted_logits[offset + j]);
    BlockScan(temp_storage).InclusiveSum(logit, prefix_sum, prefix_op);
    prefix_sum /= topp;
    if (prefix_sum >= random_n) {
      atomicMin(&result_idx, j);
    }
  }
  indices_ptr[batch_idx] = sorted_idx[offset + result_idx];

  // if (threadIdx.x == 0) {
  //   printf("selected idx: %d, %d\n", blockIdx.x, result_idx);
  // }
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
  size_t temp_storage_bytes = m->temp_storage_bytes;
  checkCUDA(cub::DeviceSegmentedRadixSort::SortPairsDescending(
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
      stream));
  int parallelism = batch_size;
  init_random_kernel<<<GET_BLOCKS(parallelism),
                       min(CUDA_NUM_THREADS, parallelism),
                       0,
                       stream>>>(m->state, batch_size, rand());
  // sampling
  sampling_topp_kernel<DT, SamplingNumThreads>
      <<<batch_size, SamplingNumThreads, 0, stream>>>(
          batch_size,
          length,
          m->state,
          static_cast<DT *>(m->sorted_logits),
          m->sorted_idx,
          indices_ptr,
          top_p);
}

/*static*/
void Sampling::forward_kernel_wrapper(SamplingMeta const *m,
                                      GenericTensorAccessorW const &input,
                                      GenericTensorAccessorW const &indices,
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
                           int total_ele,
                           GenericTensorAccessorW input,
                           MemoryAllocator &gpu_mem_allocator)
    : OpMeta(handler, op) {
  DataType data_type = op->data_type;

  size_t begin_offset_size, end_offset_size;
  begin_offset_size = end_offset_size = batch_size + 1;
  size_t idx_size, sorted_idx_size, sorted_logits_size;
  idx_size = sorted_idx_size = sorted_logits_size = total_ele;
  size_t state_size = batch_size;

  size_t totalSize = sizeof(int) * (begin_offset_size + end_offset_size +
                                    idx_size + sorted_idx_size) +
                     data_type_size(data_type) * sorted_logits_size +
                     sizeof(curandState) * state_size;
  gpu_mem_allocator.create_legion_instance(reserveInst, totalSize);
  begin_offset = gpu_mem_allocator.allocate_instance<int>(begin_offset_size);
  end_offset = gpu_mem_allocator.allocate_instance<int>(end_offset_size);
  idx = gpu_mem_allocator.allocate_instance<int>(idx_size);
  sorted_idx = gpu_mem_allocator.allocate_instance<int>(sorted_idx_size);
  sorted_logits = gpu_mem_allocator.allocate_instance_untyped(
      sorted_logits_size * data_type_size(data_type));
  state = gpu_mem_allocator.allocate_instance<curandState>(state_size);
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  // init offset
  int parallelism = total_ele;
  init_idxs<<<GET_BLOCKS(parallelism),
              min(CUDA_NUM_THREADS, parallelism),
              0,
              stream>>>(batch_size,
                        total_ele / batch_size,
                        total_ele,
                        idx,
                        begin_offset,
                        end_offset);

  // init sort function
  if (data_type == DT_FLOAT) {
    checkCUDA(cub::DeviceSegmentedRadixSort::SortPairsDescending(
        d_temp_storage,
        temp_storage_bytes,
        input.get_float_ptr(),
        input.get_float_ptr(),
        idx,
        idx,
        total_ele,
        batch_size,
        begin_offset,
        end_offset + 1,
        0,                             // begin_bit
        data_type_size(data_type) * 8, // end_bit = sizeof(KeyT) * 8
        stream));
  } else if (data_type == DT_HALF) {
    checkCUDA(cub::DeviceSegmentedRadixSort::SortPairsDescending(
        d_temp_storage,
        temp_storage_bytes,
        input.get_half_ptr(),
        input.get_half_ptr(),
        idx,
        idx,
        total_ele,
        batch_size,
        begin_offset,
        end_offset + 1,
        0,                             // begin_bit
        data_type_size(data_type) * 8, // end_bit = sizeof(KeyT) * 8
        stream));
  } else {
    assert(false && "input type in float and half");
  }

  gpu_mem_allocator.create_legion_instance(reserveInst, temp_storage_bytes);
  d_temp_storage =
      gpu_mem_allocator.allocate_instance_untyped(temp_storage_bytes);
}

SamplingMeta::~SamplingMeta(void) {
  if (reserveInst != Realm::RegionInstance::NO_INST) {
    reserveInst.destroy();
  }
}
}; // namespace FlexFlow