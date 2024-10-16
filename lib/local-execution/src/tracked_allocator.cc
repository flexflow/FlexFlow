#include "local-execution/tracked_allocator.h"
#include "kernels/device.h"

namespace FlexFlow {

TrackedAllocator::TrackedAllocator(Allocator a) : allocator(a) {}

void *TrackedAllocator::allocate(size_t requested_memory_size) {
  void *ptr = this->allocator.allocate(requested_memory_size);
  this->ptr_mem_usage.insert({ptr, requested_memory_size});
  this->current_mem_usage += requested_memory_size;
  return ptr;
}

void TrackedAllocator::deallocate(void *ptr) {
  size_t psize;
  this->ptr_mem_usage.erase(ptr);
  this->allocator.deallocate(ptr);
  this->current_mem_usage -= psize;
}

size_t TrackedAllocator::get_current_mem_usage() {
  return this->current_mem_usage;
}

DeviceType TrackedAllocator::get_allocation_device_type() const {
  return this->allocator.get_allocation_device_type();
}

Allocator get_tracked_memory_allocator(Allocator const &base_allocator) {
  Allocator allocator = Allocator::create<TrackedAllocator>(base_allocator);
  return allocator;
}

} // namespace FlexFlow
