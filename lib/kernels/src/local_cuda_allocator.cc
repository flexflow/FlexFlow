#include "kernels/local_cuda_allocator.h"
#include "kernels/device.h"

namespace FlexFlow {
void *LocalCudaAllocator::allocate(size_t requested_memory_size) {
  void *ptr;
  checkCUDA(cudaMalloc(&ptr, requested_memory_size));
  this->ptrs.insert(ptr);
  return ptr;
}

void LocalCudaAllocator::deallocate(void *ptr) {
  checkCUDA(cudaFree(ptr));
  this->ptrs.erase(ptr);
}

LocalCudaAllocator::~LocalCudaAllocator() {
  for (auto it = this->ptrs.begin(); it != this->ptrs.end();) {
    void *ptr = *it;
    it++;
    this->deallocate(ptr);
  }
}

Allocator get_local_cuda_memory_allocator() {
  return Allocator::create<LocalCudaAllocator>();
}

} // namespace FlexFlow
