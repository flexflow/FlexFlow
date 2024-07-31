#include "local-execution/tracked_allocator.h"
#include "kernels/device.h"

namespace FlexFlow {

TrackedAllocator::TrackedAllocator(Allocator a) : allocator(a) {}

void *TrackedAllocator::allocate(size_t requested_memory_size) {
  void *ptr = this->allocator.allocate(requested_memory_size);
  this->current_mem_usage += requested_memory_size;
  return ptr;
}

void *TrackedAllocator::allocate_and_zero(size_t requested_memory_size) {
  void *ptr = this->allocator.allocate_and_zero(requested_memory_size);
  this->current_mem_usage += requested_memory_size;
  return ptr;
}

void TrackedAllocator::deallocate(void *ptr) {
  size_t psize;
  checkCUDA(cudaGetSymbolSize(&psize, ptr));
  this->allocator.deallocate(ptr);
  this->current_mem_usage -= psize;
}

size_t TrackedAllocator::get_current_mem_usage() {
  return this->current_mem_usage;
}

Allocator get_tracked_memory_allocator(Allocator const &base_allocator) {
  Allocator allocator = Allocator::create<TrackedAllocator>(base_allocator);
  allocator.alloc_location = base_allocator.alloc_location;
  return allocator;
}

} // namespace FlexFlow
