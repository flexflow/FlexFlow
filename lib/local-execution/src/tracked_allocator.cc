#include "tracked_allocator.h"
#include "kernels/device.h"

namespace FlexFlow {

<<<<<<< op-refactor
void *TrackedAllocator::allocate(size_t requested_memory_size) {
  void *ptr;
  checkCUDA(cudaMalloc(&ptr, requested_memory_size));
=======
TrackedAllocator::TrackedAllocator(Allocator a) : allocator(a) {}

void *TrackedAllocator::allocate(size_t requested_memory_size) {
  void *ptr = this->allocator.allocate(requested_memory_size);
>>>>>>> repo-refactor
  this->current_mem_usage += requested_memory_size;
  return ptr;
}

void TrackedAllocator::deallocate(void *ptr) {
  size_t psize;
  checkCUDA(cudaGetSymbolSize(&psize, ptr));
<<<<<<< op-refactor
  checkCUDA(cudaFree(ptr));
=======
  this->allocator.deallocate(ptr);
>>>>>>> repo-refactor
  this->current_mem_usage -= psize;
}

size_t TrackedAllocator::get_current_mem_usage() {
  return this->current_mem_usage;
}

<<<<<<< op-refactor
Allocator get_tracked_memory_allocator() {
  return Allocator::create<TrackedAllocator>();
=======
Allocator get_tracked_memory_allocator(Allocator const &base_allocator) {
  return Allocator::create<TrackedAllocator>(base_allocator);
>>>>>>> repo-refactor
}

} // namespace FlexFlow
