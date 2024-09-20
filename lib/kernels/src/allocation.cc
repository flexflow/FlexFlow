#include "kernels/allocation.h"
#include "op-attrs/tensor_shape.h"

namespace FlexFlow {

void *Allocator::allocate(size_t mem_size) {
  return this->i_allocator->allocate(mem_size);
}

void *Allocator::allocate_and_zero(size_t mem_size) {
  return this->i_allocator->allocate_and_zero(mem_size);
}

void Allocator::deallocate(void *ptr) {
  this->i_allocator->deallocate(ptr);
}

GenericTensorAccessorW
    Allocator::allocate_tensor(TensorShape const &tensor_shape) {
  void *ptr = this->allocate(get_size_in_bytes(tensor_shape));
  bool on_device = this->alloc_location == AllocLocation::DEVICE;
  return {tensor_shape.data_type, tensor_shape, ptr, on_device};
}

GenericTensorAccessorW
    Allocator::allocate_tensor_and_zero(TensorShape const &tensor_shape) {
  void *ptr = this->allocate_and_zero(get_size_in_bytes(tensor_shape));
  bool on_device = this->alloc_location == AllocLocation::DEVICE;
  return {tensor_shape.data_type, tensor_shape, ptr, on_device};
}

} // namespace FlexFlow
