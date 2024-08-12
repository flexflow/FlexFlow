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

using Legion::Machine;
using Legion::Memory;
using Legion::Processor;

Memory get_proc_mem(Machine machine, Processor proc) {
    // First try to allocate a managed memory (cudaMallocManaged)
    Machine::MemoryQuery proc_mem = Machine::MemoryQuery(machine)
                        .only_kind(Memory::GPU_MANAGED_MEM)
                        .has_affinity_to(proc);
    if (proc_mem.count() > 0) {
      return proc_mem.first();
    }
    // If managed memory is not available, try to allocate a framebuffer memory
    proc_mem = Machine::MemoryQuery(machine)
                        .only_kind(Memory::GPU_FB_MEM)
                        .has_affinity_to(proc);
    assert(proc_mem.count() > 0);
    return proc_mem.first();
}

} // namespace FlexFlow
                        