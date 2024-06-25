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
  auto it = this->ptrs.find(ptr);
  if (it != this->ptrs.end()) {
    checkCUDA(cudaFree(ptr));
    this->ptrs.erase(ptr);
  } else {
    throw std::runtime_error(
        "Deallocating a pointer that was not allocated by this Allocator");
  }
}

LocalCudaAllocator::~LocalCudaAllocator() {
  for (auto it = this->ptrs.begin(); it != this->ptrs.end();) {
    void *ptr = *it;
    it++;
    this->deallocate(ptr);
  }
}

Allocator create_local_cuda_memory_allocator() {
  return Allocator::create<LocalCudaAllocator>();
}

} // namespace FlexFlow
