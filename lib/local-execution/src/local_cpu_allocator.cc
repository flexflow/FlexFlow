#include "local-execution/local_cpu_allocator.h"

namespace FlexFlow {
void *LocalCPUAllocator::allocate(size_t requested_memory_size) {
  void *ptr = malloc(requested_memory_size);
  this->ptrs.insert(ptr);
  return ptr;
}

void LocalCPUAllocator::deallocate(void *ptr) {
  if (contains(this->ptrs, ptr)) {
    this->ptrs.erase(ptr);
    free(ptr);
  } else {
    throw std::runtime_error(
        "Deallocating a pointer that was not allocated by this Allocator");
  }
}

LocalCPUAllocator::~LocalCPUAllocator() {
  while (!ptrs.empty()) {
    auto it = ptrs.begin();
    void *ptr = *it;
    this->deallocate(ptr);
  }
}

Allocator create_local_cpu_memory_allocator() {
  return Allocator::create<LocalCPUAllocator>();
}

} // namespace FlexFlow
