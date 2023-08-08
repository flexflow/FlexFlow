#include "kernels/allocation.h"

namespace FlexFlow {

void * Allocator::allocate(size_t size) {
  return i_allocator->allocate(size);
}

void Allocator::deallocate(void *ptr) {
  i_allocator->deallocate(ptr);
}

} // namespace FlexFlow