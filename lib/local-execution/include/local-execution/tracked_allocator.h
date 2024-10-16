#ifndef _FLEXFLOW_LOCAL_EXECUTION_TRACKED_ALLOCATOR_H
#define _FLEXFLOW_LOCAL_EXECUTION_TRACKED_ALLOCATOR_H

#include "kernels/allocation.h"

namespace FlexFlow {

struct TrackedAllocator : public IAllocator {
  TrackedAllocator(Allocator);
  TrackedAllocator(TrackedAllocator const &) = delete;
  TrackedAllocator(TrackedAllocator &&) = delete;
  ~TrackedAllocator() = default;

  void *allocate(size_t) override;
  void deallocate(void *) override;

  DeviceType get_allocation_device_type() const override;

  size_t get_current_mem_usage();

private:
  size_t current_mem_usage = 0;
  std::unordered_map<void *, size_t> ptr_mem_usage;
  Allocator allocator;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(TrackedAllocator);

Allocator get_tracked_memory_allocator(Allocator const &base_allocator);
Allocator get_tracked_local_memory_allocator();
size_t get_tracked_memory_usage(Allocator &wrapped_allocator);

} // namespace FlexFlow

#endif
