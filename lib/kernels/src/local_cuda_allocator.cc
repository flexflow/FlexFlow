#include "kernels/local_cuda_allocator.h"
#include "kernels/device.h"
#include "utils/containers/contains.h"

namespace FlexFlow {
void *LocalCudaAllocator::allocate(size_t requested_memory_size) {
  void *ptr;
  checkCUDA(cudaMalloc(&ptr, requested_memory_size));
  this->ptrs.insert(ptr);
  return ptr;
}

void LocalCudaAllocator::deallocate(void *ptr) {
  if (contains(this->ptrs, ptr)) {
    checkCUDA(cudaFree(ptr));
    this->ptrs.erase(ptr);
  } else {
    throw std::runtime_error(
        "Deallocating a pointer that was not allocated by this Allocator");
  }
}

DeviceType LocalCudaAllocator::get_allocation_device_type() const {
  return DeviceType::GPU;
}

LocalCudaAllocator::~LocalCudaAllocator() {
  for (void *ptr : this->ptrs) {
    checkCUDA(cudaFree(ptr));
  }
}

Allocator create_local_cuda_memory_allocator() {
  Allocator allocator = Allocator::create<LocalCudaAllocator>();
  return allocator;
}

} // namespace FlexFlow
