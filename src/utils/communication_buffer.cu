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

#include <string>
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
// P.S. allgather_src should be a device buffer of size CUDA_IPC_HANDLE_SIZE,
// and allgather_dst should be a device buffer of size CUDA_IPC_HANDLE_SIZE * num_devices.
std::vector<cudaIpcMemHandle_t> allgather_ipc_handles(int num_devices, ncclComm_t ncclComm,
                                                  void* allgather_src, void* allgather_dst,
                                                  cudaIpcMemHandle_t local_handle, cudaStream_t stream) {
  // int device = 0;
  // checkCUDA(cudaGetDevice(&device));
  // unsigned long long streamId;
  // checkCUDA(cudaStreamGetId(stream, &streamId));
  // printf("device %d: device_id = %d, stream = %llu\n", device, device_id, streamId);
  // fflush(stdout);

  // Copy local handle to allgather source
  checkCUDA(cudaMemcpyAsync(allgather_src, local_handle.reserved,
                            CUDA_IPC_HANDLE_SIZE, cudaMemcpyHostToDevice, stream));

  // Perform ncclAllGather to exchange handles
  checkNCCL(ncclAllGather(allgather_src, allgather_dst,
                          CUDA_IPC_HANDLE_SIZE, ncclChar, ncclComm, stream));
  // printf("device %d: allgather ipc handles done\n", device);
  // fflush(stdout);

  // Create vector to store gathered handles
  std::vector<cudaIpcMemHandle_t> handles(num_devices);
  checkCUDA(cudaMemcpyAsync(handles.data(), allgather_dst,
                            CUDA_IPC_HANDLE_SIZE * num_devices,
                            cudaMemcpyDeviceToHost, stream));
  checkCUDA(cudaStreamSynchronize(stream));

  return handles;
}

// Given a local CUDA data pointer, return the IPC memory pointers group.
// For the i-th pointer, if i is the worker id of the given device,
// then the returned i-th ptr_group is the local pointer,
// or otherwise it is an IPC memory pointer.
std::vector<void*> create_ipc_ptr_group(int num_devices, int device_id, ncclComm_t ncclComm,
                                        void* allgather_src, void* allgather_dst,
                                        void* local_ptr, cudaStream_t stream) {
  // Create ipc handle
  checkCUDA(cudaSetDevice(device_id));
  cudaIpcMemHandle_t local_handle;
  checkCUDA(cudaIpcGetMemHandle(&local_handle, local_ptr));
  // std::string str = "device " + std::to_string(device_id) + " handle: ";
  // for (int i = 0; i < CUDA_IPC_HANDLE_SIZE / sizeof(char); i++) {
  //   str += std::to_string(local_handle.reserved[i]);
  //   str += " ";
  // }
  // printf("%s\n", str.c_str());
  // fflush(stdout);
  // All-gather IPC handles.
  std::vector<cudaIpcMemHandle_t> handles = allgather_ipc_handles(num_devices, ncclComm,
                                                                  allgather_src, allgather_dst,
                                                                  local_handle, stream);
  // pid_t pid = getpid();
  // pid_t tid = syscall(SYS_gettid); // Get the thread ID
  // printf("device %d: pid = %d, tid = %d\n", device_id, pid, tid);
  // fflush(stdout);
  // while (true) {
  //   sleep(1);
  // }
  // Collect the all-gather results.
  std::vector<void*> ptr_group(num_devices);
  for (size_t node_id = 0; node_id < handles.size(); ++node_id) {
    if (static_cast<int>(node_id) == device_id) {
      ptr_group[node_id] = local_ptr;
    } else {
      void* foreign_buffer;
      // std::string str = "device " + std::to_string(device_id) + ", node " + std::to_string(node_id) + ": ";
      // for (int i = 0; i < CUDA_IPC_HANDLE_SIZE / sizeof(char); i++) {
      //   str += std::to_string(handles[node_id].reserved[i]);
      //   str += " ";
      // }
      // printf("%s\n", str.c_str());
      // fflush(stdout);
      checkCUDA(cudaIpcOpenMemHandle(&foreign_buffer, handles[node_id], cudaIpcMemLazyEnablePeerAccess));
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
// The allgather_src and allgather_dst are device buffers,
// which are used for all-gathering IPC handles across devices.
// The size of allgather_src should be CUDA_IPC_HANDLE_SIZE, and the size of allgather_dst
// should be CUDA_IPC_HANDLE_SIZE * num_devices.
CommunicationBuffer* create_comm_buf_with_local_ptr(int num_devices, int device_id, ncclComm_t ncclComm,
                                                  void* allgather_src, void* allgather_dst,
                                                  void* local_ptr, cudaStream_t stream) {
  assert(local_ptr != nullptr && "Local pointer is nullptr.");
  CommunicationBuffer* comm_buf = new CommunicationBuffer();
  comm_buf->num_devices = num_devices;
  comm_buf->device_id = device_id;
  comm_buf->local_ptr = local_ptr;
  comm_buf->comm_ptrs = create_ipc_ptr_group(num_devices, device_id, ncclComm, allgather_src, allgather_dst, local_ptr, stream);
  // printf("ipc ptr group 0 created\n");
  // fflush(stdout);

  // Create barrier helpers.
  int barrier_ptr_size = sizeof(uint32_t) * (MAX_ALL_REDUCE_BLOCKS + 2) * MAX_RANKS_PER_NODE;
  // Alloc local buffer
  void* barrier_in_ptr;
  checkCUDA(cudaMalloc(&barrier_in_ptr, barrier_ptr_size));
  checkCUDA(cudaMemset(barrier_in_ptr, 0, barrier_ptr_size));
  void* barrier_out_ptr;
  checkCUDA(cudaMalloc(&barrier_out_ptr, barrier_ptr_size));
  checkCUDA(cudaMemset(barrier_out_ptr, 0, barrier_ptr_size));
  // Reset allocated memory to zero.
  // We explicitly synchronize after memset, to make sure memset finishes
  // before using all-gather to exchange IPC handles.
  // This is important to ensure the memory reset get ordered
  // before any other peers read the memory.
  checkCUDA(cudaDeviceSynchronize());
  comm_buf->barrier_in = create_ipc_ptr_group(num_devices, device_id, ncclComm, allgather_src, allgather_dst, barrier_in_ptr, stream);
  // printf("ipc ptr group 1 created\n");
  // fflush(stdout);
  comm_buf->barrier_out = create_ipc_ptr_group(num_devices, device_id, ncclComm, allgather_src, allgather_dst, barrier_out_ptr, stream);
  // printf("ipc ptr group 2 created\n");
  // fflush(stdout);
  comm_buf->barrier_flag = 1;

  return comm_buf;
}

// Release the CommunicationBuffer.
void release_comm_buf(CommunicationBuffer* comm_buf) {
  free_ipc_ptr_group(comm_buf->comm_ptrs, comm_buf->device_id, false);
  // The local ptr of barrier_in should be freed,
  // because it is allocated in CommunicationBuffer,
  // not handled by FlexFlow runtime.
  free_ipc_ptr_group(comm_buf->barrier_in, comm_buf->device_id, true);
  free_ipc_ptr_group(comm_buf->barrier_out, comm_buf->device_id, true);
  delete comm_buf;
}
