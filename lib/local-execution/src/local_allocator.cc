#include "local_allocator.h"
#include "kernels/device.h"

namespace FlexFlow {

void *LocalAllocator::allocate(size_t requested_memory_size) {
  void *ptr;
  checkCUDA(cudaMalloc(&ptr, requested_memory_size));
  this->ptrs.insert(ptr);
  return ptr;
}

void LocalAllocator::deallocate(void *ptr) {
  checkCUDA(cudaFree(ptr));
  this->ptrs.erase(ptr);
}

LocalAllocator::~LocalAllocator() {
  for (auto it = this->ptrs.begin();
       it != this->ptrs.end();) {
    void *ptr = *it;
    it++;
    this->deallocate(ptr);
  }
}

Allocator get_local_memory_allocator() {
  return Allocator::create<LocalAllocator>();
}

} // namespace FlexFlow
