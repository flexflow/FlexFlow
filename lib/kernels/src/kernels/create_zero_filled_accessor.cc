#include "kernels/create_zero_filled_accessor.h"
#include "kernels/fill_tensor_accessor.h"

namespace FlexFlow {

GenericTensorAccessorW create_zero_filled_accessor_w(TensorShape const &shape,
                                                     Allocator &allocator) {
  GenericTensorAccessorW result_accessor = allocator.allocate_tensor(shape);
  fill_with_zeros(result_accessor);
  return result_accessor;
}

GenericTensorAccessorR create_zero_filled_accessor_r(TensorShape const &shape,
                                                     Allocator &allocator) {
  GenericTensorAccessorW accessor =
      create_zero_filled_accessor_w(shape, allocator);
  return read_only_accessor_from_write_accessor(accessor);
}

} // namespace FlexFlow
