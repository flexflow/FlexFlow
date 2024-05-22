#include "tracked_allocator.h"
#include "kernels/device.h"

namespace FlexFlow {

void *TrackedAllocator::allocate(size_t requested_memory_size) {
  void *ptr;
  checkCUDA(cudaMalloc(&ptr, requested_memory_size));
  this->current_mem_usage += requested_memory_size;
  return ptr;
}

void TrackedAllocator::deallocate(void *ptr) {
  size_t psize;
  checkCUDA(cudaGetSymbolSize(&psize, ptr));
  checkCUDA(cudaFree(ptr));
  this->current_mem_usage -= psize;
}

size_t TrackedAllocator::get_current_mem_usage() {
  return this->current_mem_usage;
}

Allocator get_tracked_memory_allocator() {
  return Allocator::create<TrackedAllocator>();
}

} // namespace FlexFlow
