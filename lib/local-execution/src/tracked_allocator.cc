#include "tracked_allocator.h"
#include "kernels/device.h"

namespace FlexFlow {

void *TrackedAllocator::allocate(size_t mem_size) {
  void *ptr = this->i_allocator->allocate(mem_size);
  this->curr_mem_usage += mem_size;
  return ptr;
}

void TrackedAllocator::deallocate(void *ptr) {
  size_t psize;
  checkCUDA(cuMemGetAddressRange(nullptr, &psize, ptr));
  this->i_allocator->deallocate(ptr);
  this->curr_mem_usage -= psize;
}

size_t TrackedAllocator::get_current_mem_usage() {
  return this->curr_mem_usage;
}

TrackedAllocator get_tracked_local_allocator() {
  return Allocator::create<LocalAllocator>();
}

} // namespace FlexFlow
