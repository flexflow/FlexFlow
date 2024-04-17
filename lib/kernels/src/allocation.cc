#include "kernels/allocation.h"

namespace FlexFlow {

void *Allocator::allocate(size_t mem_size) {
  return this->i_allocator->allocate(mem_size);
}

void Allocator::deallocate(void *ptr) {
  this->i_allocator->deallocate(ptr);
}

} // namespace FlexFlow
