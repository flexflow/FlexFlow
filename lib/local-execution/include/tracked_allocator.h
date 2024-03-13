#ifndef _FLEXFLOW_RUNTIME_SRC_LOCAL_ALLOCATOR_H
#define _FLEXFLOW_RUNTIME_SRC_LOCAL_ALLOCATOR_H

#include "kernels/allocation.h"
#include <unordered_set>

namespace FlexFlow {

struct TrackedAllocator : public IAllocator {
  TrackedAllocator(size_t) override;
  ~TrackedAllocator() override;

  void *allocate(size_t) override;
  void deallocate(void *) override;
  size_t get_ptr_memory_size(void *);

private:
  std::unordered_map<void *, size_t> ptr_memory_size_mapping;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(TrackedAllocator);

Allocator get_tracked_memory_allocator(size_t total_memory_size);

} // namespace FlexFlow

#endif
