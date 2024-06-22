#ifndef _FLEXFLOW_LOCAL_EXECUTION_TRACKED_ALLOCATOR_H
#define _FLEXFLOW_LOCAL_EXECUTION_TRACKED_ALLOCATOR_H

#include "kernels/local_allocator.h"

namespace FlexFlow {

struct TrackedAllocator : public IAllocator {
  TrackedAllocator(Allocator);
  TrackedAllocator(TrackedAllocator const &) = delete;
  TrackedAllocator(TrackedAllocator &&) = delete;
  ~TrackedAllocator() = default;

  void *allocate(size_t) override;
  void deallocate(void *) override;
  size_t get_current_mem_usage();

private:
  size_t current_mem_usage = 0;
  Allocator allocator;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(TrackedAllocator);

Allocator get_tracked_memory_allocator(Allocator const &base_allocator);
Allocator get_tracked_local_memory_allocator();
size_t get_tracked_memory_usage(Allocator &wrapped_allocator);

} // namespace FlexFlow

#endif
