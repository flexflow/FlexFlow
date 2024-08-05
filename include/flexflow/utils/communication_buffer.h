/* Copyright 2023 CMU, Stanford, Facebook, LANL
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

#ifndef _COMMUNICATION_BUFFER_H
#define _COMMUNICATION_BUFFER_H

#include <vector>
#ifdef FF_USE_NCCL
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
#include <nccl.h>
#else
#include <rccl/rccl.h>
#endif
#endif

// adapted from https://github.com/mlc-ai/relax

// The CUDA interprocess communication memory object,
// which internally contains data pointers to CUDA IPC memory.
// It is be useful for efficient all-reduce implementation.
// Right now the class members are closely tied with customized
// all-reduce kernel. They may also be extended for other uses in
// the future.
class CommunicationBuffer {
 public:
  // The device information for CUDA CommunicationBuffer.
  int num_devices;
  int device_id;
  void* local_ptr;

  // The data pointers of all all-reduce inputs.
  // It has "num_devices" pointers. The i-th pointer is the data pointer on worker i.
  // If "i != device_id", the pointer is an IPC data pointer.
  // Otherwise, the pointer is a local CUDA data pointer.
  std::vector<void*> comm_ptrs;

  // The barrier helper datas per CommunicationBuffer, which can be used 
  // by custom collective operations and allow fine-grained synchronization on each buffer.
  // They have "num_devices" pointers, and the pointer arrangement is the same as "comm_ptrs".
  std::vector<void*> barrier_in;
  std::vector<void*> barrier_out;

  // The integer buffer flag for all-reduce.
  // It will self increment by 1 after each all-reduce operation.
  int barrier_flag;
};

CommunicationBuffer* create_comm_buf_with_local_ptr(int num_devices, int device_id, ncclComm_t ncclComm,
                                                  void* local_ptr, cudaStream_t stream);

void release_comm_buf(CommunicationBuffer* comm_buf);

#endif // _COMMUNICATION_BUFFER_H
