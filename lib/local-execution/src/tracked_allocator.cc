#include "local-execution/tracked_allocator.h"
#include "kernels/device.h"

namespace FlexFlow {

TrackedAllocator::TrackedAllocator(Allocator a) : allocator(a) {}

void *TrackedAllocator::allocate(size_t requested_memory_size) {
  void *ptr = this->allocator.allocate(requested_memory_size);
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
  return Allocator::create<TrackedAllocator>(base_allocator);
}

Allocator get_tracked_local_memory_allocator() {
  return get_tracked_memory_allocator(get_local_memory_allocator());
}

size_t get_tracked_memory_usage(Allocator &wrapped_allocator) {
  TrackedAllocator &tracked_allocator =
      Allocator::unwrap<TrackedAllocator>(wrapped_allocator);
  return tracked_allocator.get_current_mem_usage();
}

} // namespace FlexFlow
