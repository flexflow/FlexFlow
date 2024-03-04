#ifndef _FLEXFLOW_RUNTIME_SRC_LOCAL_ALLOCATOR_H
#define _FLEXFLOW_RUNTIME_SRC_LOCAL_ALLOCATOR_H

#include "kernels/allocation.h"
#include <unordered_set>

namespace FlexFlow {

struct TrackedAllocator : public IAllocator {
  TrackedAllocator() = default;
  ~TrackedAllocator() override;

  void *allocate(size_t) override;
  void deallocate(void *) override;
  size_t get_ptr_memory_size(void *);

private:
  std::unordered_map<void*, size_t> ptr_memory_size_mapping;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(LocalAllocator);

Allocator get_local_memory_allocator();

} // namespace FlexFlow

#endif
