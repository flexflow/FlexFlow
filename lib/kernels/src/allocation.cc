#include "kernels/allocation.h"

namespace FlexFlow {

void *Allocator::allocate(size_t mem_size) {
  return this->i_allocator->allocate(mem_size);
}

void Allocator::deallocate(void *ptr) {
  this->i_allocator->deallocate(ptr);
}

GenericTensorAccessorW Allocator::allocate(TensorShape const &tensor_shape) {
  void *ptr = this->allocate(tensor_shape.get_volume());
  return {tensor_shape.data_type, tensor_shape, ptr};
}

} // namespace FlexFlow
