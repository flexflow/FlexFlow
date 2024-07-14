#include "kernels/local_cpu_allocator.h"
#include "kernels/device.h"

namespace FlexFlow {
void *LocalCPUAllocator::allocate(size_t requested_memory_size) {
  void *ptr = calloc(1, requested_memory_size);

  if (ptr != nullptr) {
    this->ptrs.insert(ptr);
  } else {
    throw std::bad_alloc();
  }

  return ptr;
}

void LocalCPUAllocator::deallocate(void *ptr) {
  if (contains(this->ptrs, ptr)) {
    free(ptr);
    this->ptrs.erase(ptr);
  } else {
    throw std::runtime_error(
        "Deallocating a pointer that was not allocated by this Allocator");
  }
}

LocalCPUAllocator::~LocalCPUAllocator() {
  for (auto ptr : ptrs) {
    free(ptr);
  }
}

Allocator create_local_cpu_memory_allocator() {
  return Allocator::create<LocalCPUAllocator>();
}

} // namespace FlexFlow
