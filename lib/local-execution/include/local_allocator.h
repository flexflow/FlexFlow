<<<<<<< op-refactor
#ifndef _FLEXFLOW_LOCAL_EXECUTION_LOCAL_ALLOCATOR_H
#define _FLEXFLOW_LOCAL_EXECUTION_LOCAL_ALLOCATOR_H
=======
#ifndef _FLEXFLOW_RUNTIME_SRC_LOCAL_ALLOCATOR_H
#define _FLEXFLOW_RUNTIME_SRC_LOCAL_ALLOCATOR_H
>>>>>>> repo-refactor

#include "kernels/allocation.h"
#include <unordered_set>

namespace FlexFlow {

struct LocalAllocator : public IAllocator {
  LocalAllocator() = default;
  LocalAllocator(LocalAllocator const &) = delete;
  LocalAllocator(LocalAllocator &&) = delete;
  ~LocalAllocator() = default;

  void *allocate(size_t) override;
  void deallocate(void *) override;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(LocalAllocator);

Allocator get_local_memory_allocator();

} // namespace FlexFlow

#endif
