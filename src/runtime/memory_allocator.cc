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

#include "flexflow/utils/memory_allocator.h"

namespace FlexFlow {

// declare Legion names
using Legion::coord_t;
using Legion::Memory;
using Realm::RegionInstance;

MemoryAllocator::MemoryAllocator(Memory _memory,
                                 bool _enforce_sequential_allocation)
  : memory(_memory), base_ptr(nullptr), total_size(0), allocated_size(0),
    enforce_sequential_allocation(_enforce_sequential_allocation) {}

void MemoryAllocator::allocate(RegionInstance &inst, size_t size) {
  if (enforce_sequential_allocation) {
    assert(total_size == allocated_size);
  }
  Realm::Rect<1, coord_t> bounds(Realm::Point<1, coord_t>(0),
                                 Realm::Point<1, coord_t>(total_size - 1));
  std::vector<size_t> field_sizes;
  field_sizes.push_back(sizeof(char));
  Realm::RegionInstance::create_instance(inst,
                                         memory,
                                         bounds,
                                         field_sizes,
                                         0,
                                         Realm::ProfilingRequestSet())
      .wait();
  base_ptr = inst.pointer_untyped(0, 0);
  total_size = size;
  allocated_size = 0;
}

void MemoryAllocator::allocate(void* base, size_t size)
{
  if (enforce_sequential_allocation) {
    assert(total_size == allocated_size);
  }
  base_ptr = base;
  total_size = size;
  allocated_size = 0;
}

void* MemoryAllocator::pointer_untyped(off_t offset, size_t size) {
  if (enforce_sequential_allocation) {
    assert((size_t)offset == allocated_size);
  }
  allocated_size += size;
  assert(allocated_size <= total_size);
  return static_cast<char*>(base_ptr) + offset;
}

}; // namespace FlexFlow
