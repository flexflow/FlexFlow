#include "test_utils.h"

GenericTensorAccessorR create_random_filled_accessor_r(TensorShape const &shape,
                                                       Allocator &allocator) {
  GenericTensorAccessorW accessor =
      create_random_filled_accessor_w<DataType::FLOAT>(shape, allocator);

  return read_only_accessor_from_write_accessor(accessor);
}

TensorShape make_tensor_shape_from_legion_dims(FFOrdered<size_t> dims,
                                               DataType DT) {
  return TensorShape{
      TensorDims{
          dims,
      },
      DT,
  };
}
