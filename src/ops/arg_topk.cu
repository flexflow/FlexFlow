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
#include "raft/matrix/detail/select_radix.cuh"

namespace FlexFlow {
// declare Legion names
using Legion::coord_t;

// Adopted from Raft's select_k
// https://github.com/rapidsai/raft/blob/branch-23.04/cpp/include/raft/matrix/detail/select_radix.cuh#L1113
template<typename T, typename idxT>
void raft_radix_11bits_kernel(const T* in,
                       int batch_size,
                       idxT len,
                       idxT k,
                       T* out,
                       idxT* out_idx = nullptr,
                       bool greater = true,
                       cudaStream_t stream = 0) {
    raft::matrix::detail::select::radix::select_k<T, idxT, 11, 512>(
        in,
        static_cast<idxT*>(nullptr),
        batch_size,
        len,
        k,
        out,
        out_idx,
        !greater,
        true,  // fused_last_filter
        stream);
}

// Adopted from Raft's select_k
// https://github.com/rapidsai/raft/blob/branch-23.04/cpp/include/raft/matrix/detail/select_radix.cuh#L1113
template<typename T, typename idxT>
void raft_radix_11bits_extra_pass_kernel(const T* in,
                                  int batch_size,
                                  idxT len,
                                  idxT k,
                                  T* out,
                                  idxT* out_idx = nullptr,
                                  bool greater = true,
                                  cudaStream_t stream = 0) {
    raft::matrix::detail::select::radix::select_k<T, idxT, 11, 512>(
        in,
        static_cast<idxT*>(nullptr),
        batch_size,
        len,
        k,
        out,
        out_idx,
        !greater,
        false,  // fused_last_filter
        stream);
}

__global__ void half2float_kernel(const half* __restrict__ in, float* __restrict__ out, int size) {
  // int stride = blockDim.x * gridDim.x,
  //     tid = blockIdx.x * blockDim.x + threadIdx.x;

  // for (int i = tid; i < size; i += stride) {
  //   out[i] = __half2float(in[i]);
  // }
  CUDA_KERNEL_LOOP(i, size) {
    out[i] = __half2float(in[i]);
  }
}

/*static*/
template <typename DT>
void ArgTopK::forward_kernel(
    ArgTopKMeta const *m,
    DT const *input_ptr,
    float *output_ptr,
    int *indices_ptr,
    size_t batch_size,
    int length,
    int k,
    bool sorted,
    /* Reserved: BatchConfig Updated */ BatchConfig const *bc,
    cudaStream_t stream) {
  if (m->speculative_decoding) {
    assert(bc->num_active_requests() >= 0);
    raft_radix_11bits_extra_pass_kernel<DT, int>(
        input_ptr,
        batch_size,
        length,
        BatchConfig::MAX_SPECULATIVE_TREE_BRANCHES,
        output_ptr,
        indices_ptr,
        true,
        stream);
  } else {
    // raft_radix_11bits_extra_pass_kernel<DT, int>(
    //     input_ptr,
    //     batch_size,
    //     length,
    //     k,
    //     static_cast<float*>(nullptr),
    //     indices_ptr,
    //     true,
    //     stream);
    assert(false && "Not in speculative decoding mode");
  }
}

/*static*/
void ArgTopK::forward_kernel_wrapper(ArgTopKMeta const *m,
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
    // transfer data from half to float (input to full_precision_input)
    // printf("ArgTopK: length = %d, batch_size = %d\n", length, batch_size);
    int size = length * batch_size;
    half2float_kernel<<<GET_BLOCKS(size),
                        min((int)CUDA_NUM_THREADS, size),
                        0,
                        stream>>>(input.get_half_ptr(), m->full_precision_input, size);
    ArgTopK::forward_kernel(m,
                            m->full_precision_input,
                            m->speculative_decoding ? probs.get_float_ptr()
                                                    : nullptr,
                            indices.get_int32_ptr(),
                            batch_size,
                            length,
                            k,
                            m->sorted,
                            m->speculative_decoding ? bc : nullptr,
                            stream);
  } else if (input.data_type == DT_FLOAT) {
    ArgTopK::forward_kernel(m,
                            input.get_float_ptr(),
                            m->speculative_decoding ? probs.get_float_ptr()
                                                    : nullptr,
                            indices.get_int32_ptr(),
                            batch_size,
                            length,
                            k,
                            m->sorted,
                            m->speculative_decoding ? bc : nullptr,
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
  gpu_mem_allocator.create_legion_instance(reserveInst, sizeof(float) * max_input_size);
  full_precision_input = gpu_mem_allocator.allocate_instance<float>(max_input_size);
}

ArgTopKMeta::~ArgTopKMeta() {
  if (reserveInst != Realm::RegionInstance::NO_INST) {
    reserveInst.destroy();
  }
}
}; // namespace FlexFlow
