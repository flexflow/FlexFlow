#include "kernels/allocation.h"
#include "op-attrs/tensor_shape.h"

namespace FlexFlow {

void *Allocator::allocate(size_t mem_size) {
  return this->i_allocator->allocate(mem_size);
}

void Allocator::deallocate(void *ptr) {
  this->i_allocator->deallocate(ptr);
}

GenericTensorAccessorW
    Allocator::allocate_tensor(TensorShape const &tensor_shape) {
  void *ptr = this->allocate(get_size_in_bytes(tensor_shape));
  return {tensor_shape.data_type, ArrayShape{tensor_shape}, ptr};
}

} // namespace FlexFlow
