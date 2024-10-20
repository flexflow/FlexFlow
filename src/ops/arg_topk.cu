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

#include "flexflow/ops/arg_topk.h"
#include "flexflow/utils/cuda_helper.h"
#include "raft/matrix/detail/select_k.cuh"

namespace FlexFlow {
// declare Legion names
using Legion::coord_t;

__global__ void half2float_kernel(half const *__restrict__ in,
                                  float *__restrict__ out,
                                  int size) {
  // int stride = blockDim.x * gridDim.x,
  //     tid = blockIdx.x * blockDim.x + threadIdx.x;

  // for (int i = tid; i < size; i += stride) {
  //   out[i] = __half2float(in[i]);
  // }
  CUDA_KERNEL_LOOP(i, size) {
    out[i] = __half2float(in[i]);
  }
}

template <typename DT>
__global__ void insertion_sort_kernel(DT *topk_values,
                                      int *topk_indices,
                                      int batch_size,
                                      int k) {
  int batch_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch_index < batch_size) {
    DT *values = topk_values + batch_index * k;
    int *indices = topk_indices + batch_index * k;

    for (int i = 1; i < k; i++) {
      DT key_val = values[i];
      int key_idx = indices[i];
      int j = i - 1;
      while (j >= 0 && values[j] < key_val) {
        values[j + 1] = values[j];
        indices[j + 1] = indices[j];
        j = j - 1;
      }
      values[j + 1] = key_val;
      indices[j + 1] = key_idx;
    }
  }
}

template <typename DT>
__global__ void renormalize_kernel(DT *topk_values,
                                   int batch_size,
                                   int k,
                                   float epsilon = 1e-6) {
  int batch_index = blockIdx.x * blockDim.x + threadIdx.x;
  assert(batch_index < batch_size);
  DT *values = topk_values + batch_index * k;
  DT sum = 0;
  for (int i = 0; i < k; i++) {
    sum += values[i];
  }
  sum += epsilon;
  for (int i = 0; i < k; i++) {
    values[i] /= sum;
  }
}

// Adopted from Raft's select_k
// https://github.com/rapidsai/raft/blob/branch-24.10/cpp/include/raft/matrix/detail/select_k.cuh

/*static*/
template <typename DT>
void ArgTopK::forward_kernel(
    ArgTopKMeta *m,
    DT const *input_ptr,
    DT *output_ptr,
    int *indices_ptr,
    size_t batch_size,
    int length,
    int k,
    bool sorted,
    bool renormalize,
    /* Reserved: BatchConfig Updated */ BatchConfig const *bc,
    cudaStream_t stream) {
  assert(bc->num_active_requests() >= 0);
  if (m->device_resources.find(stream) == m->device_resources.end()) {
    m->device_resources[stream] = new raft::device_resources(stream);
  }
  raft::device_resources *handle = m->device_resources[stream];
  raft::matrix::detail::select_k(*handle,
                                 input_ptr,
                                 (int *)nullptr,
                                 batch_size,
                                 (size_t)length,
                                 k,
                                 output_ptr,
                                 indices_ptr,
                                 /*select_min=*/false,
                                 sorted);
  // if (sorted) {
  //   assert(output_ptr != nullptr);
  //   insertion_sort_kernel<<<GET_BLOCKS(batch_size),
  //                           min((size_t)CUDA_NUM_THREADS, batch_size),
  //                           0,
  //                           stream>>>(output_ptr, indices_ptr, batch_size,
  //                           k);
  // }
  if (renormalize) {
    assert(output_ptr != nullptr);
    renormalize_kernel<<<GET_BLOCKS(batch_size),
                         min((size_t)CUDA_NUM_THREADS, batch_size),
                         0,
                         stream>>>(output_ptr, batch_size, k);
  }
}

/*static*/
void ArgTopK::forward_kernel_wrapper(ArgTopKMeta *m,
                                     GenericTensorAccessorR const &input,
                                     // float *output_ptr,
                                     GenericTensorAccessorW const &probs,
                                     GenericTensorAccessorW const &indices,
                                     int batch_size,
                                     BatchConfig const *bc) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  // Domain in1_domain = runtime->get_index_space_domain(
  //     ctx, task->regions[0].region.get_index_space());
  //   Domain out1_domain = runtime->get_index_space_domain(
  //       ctx, task->regions[1].region.get_index_space());
  // Domain out2_domain = runtime->get_index_space_domain(
  //     ctx, task->regions[1].region.get_index_space());
  int numdims = input.domain.get_dim();
  assert(indices.domain.get_dim() == numdims);

  int in_cols = input.domain.hi()[0] - input.domain.lo()[0] + 1;
  // int out1_cols = out1_domain.hi()[0] - out1_domain.lo()[0] + 1;
  int out2_cols = indices.domain.hi()[0] - indices.domain.lo()[0] + 1;

  // assert(out1_domain == out2_domain);
  for (int i = 1; i < input.domain.get_dim(); i++) {
    assert(input.domain.lo()[i] == indices.domain.lo()[i]);
    assert(input.domain.hi()[i] == indices.domain.hi()[i]);
  }
  // float const *in_ptr = helperGetTensorPointerRO<float>(
  //     regions[0], task->regions[0], FID_DATA, ctx, runtime);
  //   float *value_ptr = helperGetTensorPointerWO<float>(
  //       regions[1], task->regions[1], FID_DATA, ctx, runtime);
  // int *index_ptr = helperGetTensorPointerWO<int>(
  //    regions[1], task->regions[1], FID_DATA, ctx, runtime);

  int length = input.domain.hi()[0] - input.domain.lo()[0] + 1;
  int k = indices.domain.hi()[0] - indices.domain.lo()[0] +
          1; /*TODO: This prints to 5*/

  // batch_size = input.domain.get_volume() / length;
  // assert(indices.domain.get_volume() / k == batch_size);
  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  if (input.data_type == DT_HALF) {
    // printf("ArgTopK: length = %d, batch_size = %d\n", length, batch_size);
    ArgTopK::forward_kernel(m,
                            input.get_half_ptr(),
                            (half *)m->half_precision_output,
                            indices.get_int32_ptr(),
                            batch_size,
                            length,
                            m->k,
                            m->sorted,
                            m->renormalize,
                            bc,
                            stream);
    // transfer data from half to float (half_precision_output to output)
    int size = length * batch_size;
    half2float_kernel<<<GET_BLOCKS(size),
                        min((int)CUDA_NUM_THREADS, size),
                        0,
                        stream>>>(
        (half const *)m->half_precision_output, probs.get_float_ptr(), size);
  } else if (input.data_type == DT_FLOAT) {
    ArgTopK::forward_kernel(m,
                            input.get_float_ptr(),
                            probs.get_float_ptr(),
                            indices.get_int32_ptr(),
                            batch_size,
                            length,
                            m->k,
                            m->sorted,
                            m->renormalize,
                            bc,
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
    printf("[ArgTopK] forward time = %.2lfms\n", elapsed);
  }
}

ArgTopKMeta::ArgTopKMeta(FFHandler handler,
                         Op const *op,
                         MemoryAllocator &gpu_mem_allocator)
    : OpMeta(handler, op) {
  max_input_size = BatchConfig::MAX_NUM_TOKENS * 32000; // TODO: use vocab_size
  gpu_mem_allocator.create_legion_instance(reserveInst,
                                           sizeof(half) * max_input_size);
  half_precision_output = gpu_mem_allocator.allocate_instance_untyped(
      sizeof(half) * max_input_size);
}

ArgTopKMeta::~ArgTopKMeta() {
  if (reserveInst != Realm::RegionInstance::NO_INST) {
    reserveInst.destroy();
  }
  for (auto &kv : device_resources) {
    delete kv.second;
  }
}
}; // namespace FlexFlow
