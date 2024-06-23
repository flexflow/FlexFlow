#include "kernels/allocation.h"
#include <unordered_set>

namespace FlexFlow {

struct LocalAllocator : public IAllocator {
  LocalAllocator() = default;
  LocalAllocator(LocalAllocator const &) = delete;
  LocalAllocator(LocalAllocator &&) = delete;
  ~LocalAllocator() override;

  void *allocate(size_t) override;
  void deallocate(void *) override;

private:
  std::unordered_set<void *> ptrs;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(LocalAllocator);

Allocator get_local_memory_allocator();

struct LocalCPUAllocator : public IAllocator {
  LocalCPUAllocator() = default;
  LocalCPUAllocator(LocalCPUAllocator const &) = delete;
  LocalCPUAllocator(LocalCPUAllocator &&) = delete;
  ~LocalCPUAllocator() override;

  void *allocate(size_t) override;
  void deallocate(void *) override;

private:
  std::unordered_set<void *> ptrs;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(LocalCPUAllocator);

Allocator get_cpu_memory_allocator();

} // namespace FlexFlow
