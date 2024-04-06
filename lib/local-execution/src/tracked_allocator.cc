#include "tracked_allocator.h"
#include "kernels/device.h"

namespace FlexFlow {

TrackedAllocator::TrackedAllocator(size_t total_memory_size)
    : total_memory_size(total_memory_size), allocated_memory_size(0) {}

void *TrackedAllocator::allocate(size_t requested_memory_size) {
  void *ptr;
  checkCUDA(cudaMalloc(&ptr, requested_memory_size));
  this->allocated_memory_size += requested_memory_size;
  this->ptr_memory_size_mapping[ptr] = requested_memory_size;
  return ptr;
}

void TrackedAllocator::deallocate(void *ptr) {
  size_t freed_memory_size = this->ptr_memory_size_mapping[ptr];
  checkCUDA(cudaFree(ptr));
  this->allocated_memory_size -= freed_memory_size;
  this->ptr_memory_size_mapping.erase(ptr);
}

size_t TrackedAllocator::get_ptr_memory_size(void *ptr) {
  auto it = this->ptr_memory_size_mapping.find(ptr);
  if (it != this->ptr_memory_size_mapping.end()) {
    return it->second;
  } else {
    throw mk_runtime_error("Requested pointer has no associated memory");
  }
}

TrackedAllocator::~TrackedAllocator() {
  for (auto it = this->ptr_memory_size_mapping.begin();
       it != this->ptr_memory_size_mapping.end();) {
    void *ptr = it->first;
    it++;
    this->deallocate(ptr);
  }
}

Allocator get_tracked_memory_allocator(size_t total_memory_size) {
  return Allocator::create<TrackedAllocator>(total_memory_size);
}

} // namespace FlexFlow
