#include "cuda_allocator.h"

namespace FlexFlow {

void * CudaAllocator::allocate(size_t size) {
  void *ptr;
  check_CUDA(cudaMalloc(&ptr, size));
  return ptr;
}

void CudaAllocator::deallocate(void *ptr) {
  check_CUDA(cudaFree(ptr));
}

} // namespace FlexFlow


