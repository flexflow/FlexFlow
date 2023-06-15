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
  inline void *allocate_untyped(size_t datalen) {
    void *ptr = static_cast<char *>(base_ptr) + allocated_size;
    allocated_size += datalen;
    assert(allocated_size <= total_size);
    return ptr;
  }
  template <typename DT>
  inline DT *allocate(size_t count) {
    void *ptr = static_cast<char *>(base_ptr) + allocated_size;
    allocated_size += sizeof(DT) * count;
    assert(allocated_size <= total_size);
    return static_cast<DT *>(ptr);
  }

public:
  Legion::Memory memory;
  void *base_ptr;
  size_t total_size, allocated_size;
  bool use_reserved_work_space;
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_RUNTIME_H_
