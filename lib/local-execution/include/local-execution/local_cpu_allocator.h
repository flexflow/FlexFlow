#include "kernels/allocation.h"
#include <unordered_set>

namespace FlexFlow {

struct LocalCPUAllocator : public IAllocator {
  LocalCPUAllocator() = default;
  LocalCPUAllocator(LocalCPUAllocator const &) = delete;
  LocalCPUAllocator(LocalCPUAllocator &&) = delete;
  ~LocalCPUAllocator() = default;

  void *allocate(size_t) override;
  void deallocate(void *) override;

  DeviceType get_allocation_device_type() const override;

private:
  std::unordered_map<void *, std::unique_ptr<void, decltype(&free)>> ptrs;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(LocalCPUAllocator);

Allocator create_local_cpu_memory_allocator();

} // namespace FlexFlow
