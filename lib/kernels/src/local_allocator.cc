#include "kernels/local_allocator.h"
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
  for (auto it = this->ptrs.begin(); it != this->ptrs.end();) {
    void *ptr = *it;
    it++;
    this->deallocate(ptr);
  }
}

Allocator get_local_memory_allocator() {
  return Allocator::create<LocalAllocator>();
}

void *LocalCPUAllocator::allocate(size_t requested_memory_size) {
  void *ptr = malloc(requested_memory_size);
  if (ptr) {
    this->ptrs.insert(ptr);
  } else {
    throw std::bad_alloc();
  }
  return ptr;
}

void LocalCPUAllocator::deallocate(void *ptr) {
  auto it = this->ptrs.find(ptr);
  if (it != this->ptrs.end()) {
    free(ptr);
    this->ptrs.erase(it);
  } else {
    throw std::runtime_error(
        "Deallocating a pointer that was not allocated by this allocator");
  }
}

LocalCPUAllocator::~LocalCPUAllocator() {
  for (auto it = this->ptrs.begin(); it != this->ptrs.end();) {
    void *ptr = *it;
    it++;
    this->deallocate(ptr);
  }
}

Allocator get_cpu_memory_allocator() {
  return Allocator::create<LocalCPUAllocator>();
}

} // namespace FlexFlow
