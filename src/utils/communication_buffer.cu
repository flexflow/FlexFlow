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

#include <cuda_runtime.h>
#include "flexflow/utils/communication_buffer.h"
#include "flexflow/utils/cuda_helper.h"
#include "tensorrt_llm/custom_allreduce_kernels.h"

using tensorrt_llm::MAX_ALL_REDUCE_BLOCKS;
using tensorrt_llm::MAX_RANKS_PER_NODE;

// All-gather the IPC memory handles across all distributed workers.
// On each worker, we copy the IPC handle to GPU memory. And nccl AllGather
// is reused to all-gather the handles. Finally the all-gathered handles
// on each worker are copied from GPU to CPU.
std::vector<cudaIpcMemHandle_t> allgather_ipc_handles(int num_devices, int device_id, ncclComm_t ncclComm,
                                                  cudaIpcMemHandle_t local_handle, cudaStream_t stream) {
  void *d_src, *d_dst;
  checkCUDA(cudaMalloc(&d_src, CUDA_IPC_HANDLE_SIZE));
  checkCUDA(cudaMalloc(&d_dst, CUDA_IPC_HANDLE_SIZE * num_devices));
  checkCUDA(cudaMemcpy(d_src, &local_handle, CUDA_IPC_HANDLE_SIZE, cudaMemcpyHostToDevice));
  checkNCCL(
      ncclAllGather(d_src, d_dst, CUDA_IPC_HANDLE_SIZE, ncclChar, ncclComm, stream));
  std::vector<char> serial_handles(CUDA_IPC_HANDLE_SIZE * num_devices, 0);
  checkCUDA(cudaMemcpy(serial_handles.data(), d_dst,
                       CUDA_IPC_HANDLE_SIZE * num_devices, cudaMemcpyDefault));
  std::vector<cudaIpcMemHandle_t> handles(num_devices);
  for (int i = 0; i < num_devices; ++i) {
    memcpy(handles[i].reserved, &serial_handles[i * CUDA_IPC_HANDLE_SIZE], CUDA_IPC_HANDLE_SIZE);
  }
  checkCUDA(cudaFree(d_src));
  checkCUDA(cudaFree(d_dst));
  return handles;
}

// Given a local CUDA data pointer, return the IPC memory pointers group.
// For the i-th pointer, if i is the worker id of the given device,
// then the returned i-th ptr_group is the local pointer,
// or otherwise it is an IPC memory pointer.
std::vector<void*> create_ipc_ptr_group(int num_devices, int device_id, ncclComm_t ncclComm,
                                        void* local_ptr, cudaStream_t stream) {
  // Create ipc handle
  cudaIpcMemHandle_t local_handle;
  checkCUDA(cudaIpcGetMemHandle(&local_handle, local_ptr));
  // All-gather IPC handles.
  std::vector<cudaIpcMemHandle_t> handles = allgather_ipc_handles(num_devices, device_id, ncclComm,
                                                                  local_handle, stream);
  // Collect the all-gather results.
  std::vector<void*> ptr_group(num_devices);
  for (size_t node_id = 0; node_id < handles.size(); ++node_id) {
    if (static_cast<int>(node_id) == device_id) {
      ptr_group[node_id] = local_ptr;
    } else {
      uint8_t* foreign_buffer;
      checkCUDA(cudaIpcOpenMemHandle(reinterpret_cast<void**>(&foreign_buffer), handles[node_id],
                                      cudaIpcMemLazyEnablePeerAccess));
      ptr_group[node_id] = foreign_buffer;
    }
  }
  return ptr_group;
}

// Free the IPC memory pointers group.
void free_ipc_ptr_group(std::vector<void*> ptr_group, int device_id, bool free_local) {
  for (int i = 0; i < static_cast<int>(ptr_group.size()); ++i) {
    if (i != device_id) {
      // Free ipc handle.
      checkCUDA(cudaIpcCloseMemHandle(ptr_group[i]));
    } else if (free_local) {
      // Free local buffer.
      checkCUDA(cudaFree(ptr_group[i]));
    }
  }
}

// Given a local CUDA data pointer, return the CommunicationBuffer of the pointer.
// The CommunicationBuffer contains the local pointer and the IPC memory pointers group.
// It contains the barrier helpers for synchronization across distributed workers,
// which is also IPC-based.
CommunicationBuffer create_comm_buf_with_local_ptr(int num_devices, int device_id, ncclComm_t ncclComm,
                                                  void* local_ptr, cudaStream_t stream) {
  assert(local_ptr != nullptr && "Local pointer is nullptr.");
  CommunicationBuffer comm_buf;
  comm_buf.num_devices = num_devices;
  comm_buf.device_id = device_id;
  comm_buf.local_ptr = local_ptr;
  comm_buf.comm_ptrs = create_ipc_ptr_group(num_devices, device_id, ncclComm, local_ptr, stream);

  // Create barrier helpers.
  int barrier_ptr_size = sizeof(uint32_t) * (MAX_ALL_REDUCE_BLOCKS + 2) * MAX_RANKS_PER_NODE;
  // Alloc local buffer
  void* barrier_in_ptr;
  checkCUDA(cudaMalloc(&barrier_in_ptr, barrier_ptr_size));
  // Reset allocated memory to zero.
  // We explicitly synchronize after memset, to make sure memset finishes
  // before using all-gather to exchange IPC handles.
  // This is important to ensure the memory reset get ordered
  // before any other peers read the memory.
  checkCUDA(cudaMemset(barrier_in_ptr, 0, barrier_ptr_size));
  checkCUDA(cudaDeviceSynchronize());
  void* barrier_out_ptr;
  checkCUDA(cudaMalloc(&barrier_out_ptr, barrier_ptr_size));
  checkCUDA(cudaMemset(barrier_out_ptr, 0, barrier_ptr_size));
  checkCUDA(cudaDeviceSynchronize());
  comm_buf.barrier_in = create_ipc_ptr_group(num_devices, device_id, ncclComm, barrier_in_ptr, stream);
  comm_buf.barrier_out = create_ipc_ptr_group(num_devices, device_id, ncclComm, barrier_out_ptr, stream);
  comm_buf.barrier_flag = 1;

  return comm_buf;
}

// Release the CommunicationBuffer.
void release_comm_buf(CommunicationBuffer comm_buf) {
  free_ipc_ptr_group(comm_buf.comm_ptrs, comm_buf.device_id, false);
  // The local ptr of barrier_in should be freed,
  // because it is allocated in CommunicationBuffer,
  // not handled by FlexFlow runtime.
  free_ipc_ptr_group(comm_buf.barrier_in, comm_buf.device_id, true);
  free_ipc_ptr_group(comm_buf.barrier_out, comm_buf.device_id, true);
}
