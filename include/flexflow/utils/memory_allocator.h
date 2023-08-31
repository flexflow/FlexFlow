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

#ifndef _FLEXFLOW_UTILS_MEMORY_ALLOCATOR_H_
#define _FLEXFLOW_UTILS_MEMORY_ALLOCATOR_H_

#include "flexflow/config.h"

namespace FlexFlow {

class MemoryAllocator {
public:
  MemoryAllocator(Legion::Memory memory);
  void create_legion_instance(Realm::RegionInstance &inst, size_t size);
  void register_reserved_work_space(void *base, size_t size);
  inline void *allocate_reserved_untyped(size_t datalen) {
    void *ptr = static_cast<char *>(reserved_ptr) + reserved_allocated_size;
    reserved_allocated_size += datalen;
    assert(reserved_allocated_size <= reserved_total_size);
    return ptr;
  }
  template <typename DT>
  inline DT *allocate_reserved(size_t count) {
    void *ptr = static_cast<char *>(reserved_ptr) + reserved_allocated_size;
    reserved_allocated_size += sizeof(DT) * count;
    assert(reserved_allocated_size <= reserved_total_size);
    return static_cast<DT *>(ptr);
  }

  inline void *allocate_instance_untyped(size_t datalen) {
    void *ptr = static_cast<char *>(instance_ptr) + instance_allocated_size;
    instance_allocated_size += datalen;
    assert(instance_allocated_size <= instance_total_size);
    return ptr;
  }

  template <typename DT>
  inline DT *allocate_instance(size_t count) {
    void *ptr = static_cast<char *>(instance_ptr) + instance_allocated_size;
    instance_allocated_size += sizeof(DT) * count;
    assert(instance_allocated_size <= instance_total_size);
    return static_cast<DT *>(ptr);
  }

public:
  Legion::Memory memory;
  void *reserved_ptr;
  void *instance_ptr;
  size_t reserved_total_size, reserved_allocated_size;
  size_t instance_total_size, instance_allocated_size;
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_RUNTIME_H_
