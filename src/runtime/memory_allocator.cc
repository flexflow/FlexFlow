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

MemoryAllocator::MemoryAllocator(Memory _memory)
    : memory(_memory), base_ptr(nullptr), total_size(0), allocated_size(0),
      use_reserved_work_space(false) {}

void MemoryAllocator::create_legion_instance(RegionInstance &inst,
                                             size_t size) {
  // Assert that we have used up previously created region instance
  assert(total_size == allocated_size);
  Realm::Rect<1, coord_t> bounds(Realm::Point<1, coord_t>(0),
                                 Realm::Point<1, coord_t>(size - 1));
  std::vector<size_t> field_sizes;
  field_sizes.push_back(sizeof(char));
  Realm::RegionInstance::create_instance(
      inst, memory, bounds, field_sizes, 0, Realm::ProfilingRequestSet())
      .wait();
  base_ptr = inst.pointer_untyped(0, 0);
  total_size = size;
  allocated_size = 0;
}

void MemoryAllocator::register_reserved_work_space(void *base, size_t size) {
  // Assert that we haven't allocated anything before
  assert(total_size == 0);
  base_ptr = base;
  total_size = size;
  allocated_size = 0;
  use_reserved_work_space = true;
}

}; // namespace FlexFlow
