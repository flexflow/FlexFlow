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

// Given a local CUDA data pointer, return the peer memory pointers group.
// For the i-th pointer, if i is the worker id of the given device,
// then the returned i-th ptr_group is the local pointer,
// or otherwise it is an peer memory pointer from the remote device.
std::vector<void*> create_peer_ptr_group(int num_devices, int device_id, ncclComm_t ncclComm,
                                        void* allgather_src, void* allgather_dst,
                                        void* local_ptr, cudaStream_t stream) {
  // Ensure we are on the correct device
  int device = 0;
  checkCUDA(cudaGetDevice(&device));
  assert(device == device_id && "Device ID does not match current device.");

  // Next we all-gather the peer memory pointers across all distributed workers.
  // On each worker, we copy the peer pointers to GPU memory. And nccl AllGather
  // is used to all-gather the pointers. Finally the all-gathered pointers
  // on each worker are copied from GPU to CPU.

  checkCUDA(cudaMemcpyAsync(allgather_src, &local_ptr, sizeof(void*),
                            cudaMemcpyHostToDevice, stream));

  checkNCCL(ncclAllGather(allgather_src, allgather_dst,
                          sizeof(void*), ncclChar, ncclComm, stream));

  std::vector<void*> peer_pointers(num_devices);
  checkCUDA(cudaMemcpyAsync(peer_pointers.data(), allgather_dst,
                            sizeof(void*) * num_devices,
                            cudaMemcpyDeviceToHost, stream));
  checkCUDA(cudaStreamSynchronize(stream));

  return peer_pointers;
}

// Free the peer memory pointers group.
void free_peer_ptr_group(std::vector<void*> ptr_group, int device_id, bool free_local) {
    for (int i = 0; i < static_cast<int>(ptr_group.size()); ++i) {
        if (i == device_id && free_local) {
            // Free the local buffer.
            checkCUDA(cudaFree(ptr_group[i]));
        }
        // No need to do anything for other devices.
    }
}

// Given a local CUDA data pointer, return the CommunicationBuffer of the pointer.
// The CommunicationBuffer contains the local pointer and the peer memory pointers group.
// It contains the barrier helpers for synchronization across distributed workers,
// which should also be peer-based.
// The allgather_src and allgather_dst are device buffers,
// which are used for all-gathering peer pointers across devices.
// The size of allgather_src should be sizeof(void*),
// and the size of allgather_dst should be sizeof(void*) * num_devices.
CommunicationBuffer* create_comm_buf_with_local_ptr(int num_devices, int device_id, ncclComm_t ncclComm,
                                                  void* allgather_src, void* allgather_dst,
                                                  void* local_ptr, void* barrier_in_ptr, void* barrier_out_ptr,
                                                  cudaStream_t stream) {
  assert(local_ptr != nullptr && "Local pointer is nullptr.");
  CommunicationBuffer* comm_buf = new CommunicationBuffer();
  comm_buf->num_devices = num_devices;
  comm_buf->device_id = device_id;
  comm_buf->local_ptr = local_ptr;
  comm_buf->comm_ptrs = create_peer_ptr_group(num_devices, device_id, ncclComm, allgather_src, allgather_dst, local_ptr, stream);
  comm_buf->barrier_in = create_peer_ptr_group(num_devices, device_id, ncclComm, allgather_src, allgather_dst, barrier_in_ptr, stream);
  comm_buf->barrier_out = create_peer_ptr_group(num_devices, device_id, ncclComm, allgather_src, allgather_dst, barrier_out_ptr, stream);
  comm_buf->barrier_flag = 1;

  return comm_buf;
}

// Release the CommunicationBuffer.
void release_comm_buf(CommunicationBuffer* comm_buf) {
  free_peer_ptr_group(comm_buf->comm_ptrs, comm_buf->device_id, false);
  free_peer_ptr_group(comm_buf->barrier_in, comm_buf->device_id, false);
  free_peer_ptr_group(comm_buf->barrier_out, comm_buf->device_id, false);
  delete comm_buf;
}
