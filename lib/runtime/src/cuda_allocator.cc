#include "cuda_allocator.h"
#include "kernels/device.h"

namespace FlexFlow {

void *CudaAllocator::allocate(size_t size) {
  void *ptr;
  checkCUDA(cudaMalloc(&ptr, size));
  return ptr;
}

void CudaAllocator::deallocate(void *ptr) {
  checkCUDA(cudaFree(ptr));
}

} // namespace FlexFlow
