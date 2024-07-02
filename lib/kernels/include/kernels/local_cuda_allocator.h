#include "kernels/allocation.h"
#include <unordered_set>

namespace FlexFlow {

struct LocalCudaAllocator : public IAllocator {
  LocalCudaAllocator() = default;
  LocalCudaAllocator(LocalCudaAllocator const &) = delete;
  LocalCudaAllocator(LocalCudaAllocator &&) = delete;
  ~LocalCudaAllocator() override;

  void *allocate(size_t) override;
  void deallocate(void *) override;

private:
  std::unordered_set<void *> ptrs;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(LocalCudaAllocator);

Allocator create_local_cuda_memory_allocator();

} // namespace FlexFlow
