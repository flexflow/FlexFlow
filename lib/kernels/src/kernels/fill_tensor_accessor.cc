#include "kernels/fill_tensor_accessor.h"
#include "kernels/datatype_dispatch.h"
#include "op-attrs/datatype_value.h"
#include "op-attrs/tensor_shape.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

template <DataType DT>
struct FillWithZeros {
  void operator()(GenericTensorAccessorW const &accessor) {
    using T = real_type_t<DT>;

    if (accessor.device_type == DeviceType::CPU) {
      memset(accessor.ptr,
             0,
             get_size_in_bytes(accessor.shape)
                 .unwrap_num_bytes()
                 .unwrap_nonnegative());
    } else {
      checkCUDA(cudaMemset(accessor.ptr,
                           0,
                           get_size_in_bytes(accessor.shape)
                               .unwrap_num_bytes()
                               .unwrap_nonnegative()));
    }
  }
};

void fill_with_zeros(GenericTensorAccessorW const &accessor) {
  DataTypeDispatch1<FillWithZeros>{}(accessor.shape.data_type, accessor);
}

GenericTensorAccessorW create_accessor_w_filled_with(
    TensorShape const &shape, DataTypeValue val, Allocator const &allocator) {
  NOT_IMPLEMENTED();
}

GenericTensorAccessorR create_accessor_r_filled_with(
    TensorShape const &shape, DataTypeValue val, Allocator const &allocator) {
  return read_only_accessor_from_write_accessor(
      create_accessor_w_filled_with(shape, val, allocator));
}

} // namespace FlexFlow
