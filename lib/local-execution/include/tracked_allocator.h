#ifndef _FLEXFLOW_LOCAL_EXECUTION_TRACKED_ALLOCATOR_H
#define _FLEXFLOW_LOCAL_EXECUTION_TRACKED_ALLOCATOR_H

#include "kernels/allocation.h"

namespace FlexFlow {

struct TrackedAllocator : public IAllocator {
  TrackedAllocator() = default;
  TrackedAllocator(TrackedAllocator const &) = delete;
  TrackedAllocator(TrackedAllocator &&) = delete;
  ~TrackedAllocator() = default;

  void *allocate(size_t) override;
  void deallocate(void *) override;
  size_t get_current_mem_usage();

private:
  size_t current_mem_usage;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(TrackedAllocator);

Allocator get_tracked_memory_allocator();

} // namespace FlexFlow

#endif
