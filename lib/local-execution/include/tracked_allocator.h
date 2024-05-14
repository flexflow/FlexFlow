#ifndef _FLEXFLOW_LOCAL_EXECUTION_TRACKED_ALLOCATOR_H
#define _FLEXFLOW_LOCAL_EXECUTION_TRACKED_ALLOCATOR_H

#include "kernels/allocation.h"

namespace FlexFlow {

struct TrackedAllocator : public Allocator {
  Allocator() = delete;

  void *allocate(size_t mem_size);
  void deallocate(void *ptr);
  size_t get_current_mem_usage();

private:
  size_t current_mem_usage;
};

} // namespace FlexFlow

#endif
