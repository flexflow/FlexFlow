#ifndef _FLEXFLOW_RUNTIME_SRC_LOCAL_ALLOCATOR_H
#define _FLEXFLOW_RUNTIME_SRC_LOCAL_ALLOCATOR_H

#include "kernels/allocation.h"
#include <unordered_set>

namespace FlexFlow {

struct LocalAllocator : public IAllocator {
  LocalAllocator() = default;
  ~LocalAllocator() override;

  void *allocate(size_t) override;
  void deallocate(void *) override;

private:
  std::unordered_set<void*> ptrs;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(LocalAllocator);

Allocator get_local_memory_allocator();

} // namespace FlexFlow

#endif
