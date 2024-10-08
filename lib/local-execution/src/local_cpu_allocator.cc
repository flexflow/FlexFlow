#include "local-execution/local_cpu_allocator.h"
#include "utils/containers/contains_key.h"

namespace FlexFlow {
void *LocalCPUAllocator::allocate(size_t requested_memory_size) {
  void *ptr = malloc(requested_memory_size);
  this->ptrs.insert({ptr, std::unique_ptr<void, decltype(&free)>(ptr, free)});
  return ptr;
}

void LocalCPUAllocator::deallocate(void *ptr) {
  if (contains_key(this->ptrs, ptr)) {
    this->ptrs.erase(ptr);
  } else {
    throw std::runtime_error(
        "Deallocating a pointer that was not allocated by this Allocator");
  }
}

DeviceType LocalCPUAllocator::get_allocation_device_type() const {
  return DeviceType::CPU;
}

Allocator create_local_cpu_memory_allocator() {
  return Allocator::create<LocalCPUAllocator>();
}

} // namespace FlexFlow
