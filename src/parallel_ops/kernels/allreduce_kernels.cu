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

#include "flexflow/parallel_ops/kernels/allreduce_kernels.h"
#include "flexflow/utils/cuda_helper.h"
#include "tensorrt_llm/custom_allreduce_kernels.h"
#include <cuda_runtime.h>

namespace FlexFlow {

AllReduceMeta::AllReduceMeta(FFHandler handle,
                             AllReduce const *reduct,
                             MemoryAllocator &gpu_mem_allocator)
    : OpMeta(handle) {
  barrier_ptr_size = sizeof(uint32_t) *
                     (tensorrt_llm::MAX_ALL_REDUCE_BLOCKS + 2) *
                     tensorrt_llm::MAX_RANKS_PER_NODE;
  gpu_mem_allocator.create_legion_instance(
      reserveInst,
      sizeof(void *) * (handle.num_devices + 1) + barrier_ptr_size * 2);
  allgather_src = gpu_mem_allocator.allocate_instance_untyped(sizeof(void *));
  allgather_dst = gpu_mem_allocator.allocate_instance_untyped(
      sizeof(void *) * handle.num_devices);
  // Create barrier helpers for all-reduce.
  barrier_in_ptr =
      gpu_mem_allocator.allocate_instance_untyped(barrier_ptr_size);
  barrier_out_ptr =
      gpu_mem_allocator.allocate_instance_untyped(barrier_ptr_size);
  checkCUDA(cudaMemset(barrier_in_ptr, 0, barrier_ptr_size));
  checkCUDA(cudaMemset(barrier_out_ptr, 0, barrier_ptr_size));
  // Reset allocated memory to zero.
  // We explicitly synchronize after memset, to make sure memset finishes
  // before using all-gather to exchange peer pointers.
  // This is important to ensure the memory reset get ordered
  // before any other peers read the memory.
  checkCUDA(cudaDeviceSynchronize());
}

AllReduceMeta::~AllReduceMeta() {
  if (reserveInst != Realm::RegionInstance::NO_INST) {
    reserveInst.destroy();
  }
}

namespace Kernels {
namespace AllReduce {

CommunicationBuffer *get_or_create_comm_buffer(AllReduceMeta *m,
                                               int num_devices,
                                               int device_id,
                                               ncclComm_t ncclComm,
                                               void *local_ptr,
                                               cudaStream_t stream) {
  auto iter = m->comm_bufs.find(local_ptr);
  if (iter != m->comm_bufs.end()) {
    return iter->second;
  } else {
    CommunicationBuffer *comm_buffer =
        create_comm_buf_with_local_ptr(num_devices,
                                       device_id,
                                       ncclComm,
                                       m->allgather_src,
                                       m->allgather_dst,
                                       local_ptr,
                                       m->barrier_in_ptr,
                                       m->barrier_out_ptr,
                                       &(m->barrier_flag),
                                       stream);
    m->comm_bufs[local_ptr] = comm_buffer;
    return comm_buffer;
  }
}

// Get the number of bits for a given data type.
inline int get_bits(DataType dtype) {
  switch (dtype) {
    case DataType::DT_INT64:
    case DataType::DT_DOUBLE:
      return 64;
    case DataType::DT_INT32:
    case DataType::DT_FLOAT:
      return 32;
    case DataType::DT_HALF:
      return 16;
    case DataType::DT_INT8:
      return 8;
    case DataType::DT_INT4:
      return 4;
    default:
      assert(false && "Unsupported data type");
  }
}

// Check if customized all-reduce kernels can be applied.
inline bool CanApplyCustomAllReduce(int64_t num_elements, DataType dtype) {
  // The customized all-reduce kernel has the following requirement(s).
  return num_elements % (16 / ((get_bits(dtype) + 7) / 8)) == 0;
}

// Check if the two-shot customized all-reduce kernel can be applied.
inline bool CanApplyTwoShotAllReduce(int64_t num_elements,
                                     DataType dtype,
                                     int num_workers) {
  // The two-shot customized all-reduce kernel has the following requirement(s).
  return (num_elements / num_workers) % (16 / ((get_bits(dtype) + 7) / 8)) == 0;
}

// Customized all-reduce kernel backed by CUDA Peer memory.
void inference_kernel_wrapper(AllReduceMeta *m,
                              BatchConfig const *bc,
                              GenericTensorAccessorR const &input,
                              GenericTensorAccessorW const &output) {
#ifndef FF_USE_NCCL
  assert(false && "Must enable FF_USE_NCCL to use AllReduce operators");
#endif
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  assert(input.data_type == output.data_type);
  assert(input.domain == output.domain);
  size_t hidden_dim_size = input.domain.hi()[0] - input.domain.lo()[0] + 1;
  size_t num_elements = bc->num_tokens * hidden_dim_size;
  int num_devices = m->handle.num_devices;
  int device_id = m->handle.device_id;
  ncclComm_t ncclComm = m->handle.ncclComm;
  DataType dtype = input.data_type;

  tensorrt_llm::AllReduceStrategyType strategy =
      tensorrt_llm::SelectImplementation(
          num_elements * ((get_bits(dtype) + 7) / 8), num_devices);

  if (strategy == tensorrt_llm::AllReduceStrategyType::RING ||
      !CanApplyCustomAllReduce(num_elements, dtype)) {
    // Dispatch to nccl AllReduce if the customized all-reduce cannot apply.
    ncclDataType_t nccl_data_type = ff_to_nccl_datatype(dtype);
    checkNCCL(ncclAllReduce(input.ptr,
                            output.ptr,
                            num_elements,
                            nccl_data_type,
                            ncclSum,
                            ncclComm,
                            stream));
    return;
  }

  // Initialize the all-reduce kernel arguments.
  tensorrt_llm::AllReduceParams params;
  params.ranks_per_node = num_devices;
  params.rank = device_id;
  params.local_rank = device_id;
  CommunicationBuffer *comm_buffer =
      get_or_create_comm_buffer(m,
                                num_devices,
                                device_id,
                                ncclComm,
                                const_cast<void *>(input.ptr),
                                stream);
  params.barrier_flag = ++(*comm_buffer->barrier_flag);
  for (int i = 0; i < num_devices; ++i) {
    params.peer_comm_buffer_ptrs[i] = comm_buffer->comm_ptrs[i];
  }
  for (int i = 0; i < num_devices; ++i) {
    params.peer_barrier_ptrs_in[i] =
        reinterpret_cast<uint32_t *>(comm_buffer->barrier_in[i]);
  }
  for (int i = 0; i < num_devices; ++i) {
    params.peer_barrier_ptrs_out[i] =
        reinterpret_cast<uint32_t *>(comm_buffer->barrier_out[i]);
  }

  if (!CanApplyTwoShotAllReduce(num_elements, dtype, num_devices)) {
    // Two-shot all-reduce does not support this case.
    // So we fallback to the one-shot strategy.
    strategy = tensorrt_llm::AllReduceStrategyType::ONESHOT;
  }

  tensorrt_llm::customAllReduce(
      params, output.ptr, num_elements, dtype, strategy, stream);
}

void forward_kernel_wrapper(AllReduceMeta const *m,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorW const &output) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  assert(input.data_type == output.data_type);
  assert(input.domain == output.domain);
#ifdef FF_USE_NCCL
  ncclDataType_t nccl_data_type = ff_to_nccl_datatype(input.data_type);
  checkNCCL(ncclAllReduce(input.ptr,
                          output.ptr,
                          input.domain.get_volume(),
                          nccl_data_type,
                          ncclSum,
                          m->handle.ncclComm,
                          stream));
#else
  assert(false && "Must enable FF_USE_NCCL to use AllReduce operators");
#endif
}

void backward_kernel_wrapper(AllReduceMeta const *m,
                             GenericTensorAccessorW const &input_grad,
                             GenericTensorAccessorR const &output_grad) {
  assert(false && "To be implemented");
}

} // namespace AllReduce
} // namespace Kernels
} // namespace FlexFlow
