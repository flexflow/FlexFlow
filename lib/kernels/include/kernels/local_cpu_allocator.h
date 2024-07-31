#include "kernels/allocation.h"
#include <unordered_set>

namespace FlexFlow {

struct LocalCPUAllocator : public IAllocator {
  LocalCPUAllocator() = default;
  LocalCPUAllocator(LocalCPUAllocator const &) = delete;
  LocalCPUAllocator(LocalCPUAllocator &&) = delete;
  ~LocalCPUAllocator() override;

  void *allocate(size_t) override;
  void *allocate_and_zero(size_t) override;
  void deallocate(void *) override;

private:
  std::unordered_set<void *> ptrs;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(LocalCPUAllocator);

Allocator create_local_cpu_memory_allocator();

} // namespace FlexFlow
