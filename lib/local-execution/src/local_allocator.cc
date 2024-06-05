#include "local-execution/local_allocator.h"
#include "kernels/device.h"

namespace FlexFlow {

void *LocalAllocator::allocate(size_t requested_memory_size) {
  void *ptr;
  checkCUDA(cudaMalloc(&ptr, requested_memory_size));
  return ptr;
}

void LocalAllocator::deallocate(void *ptr) {
  checkCUDA(cudaFree(ptr));
}

Allocator get_local_memory_allocator() {
  return Allocator::create<LocalAllocator>();
}

} // namespace FlexFlow
