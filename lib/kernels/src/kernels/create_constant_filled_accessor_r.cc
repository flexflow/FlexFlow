#include "kernels/create_constant_filled_accessor_r.h"
#include "kernels/accessor.h"
#include "kernels/datatype_dispatch.h"
#include "kernels/local_cpu_allocator.h"
#include "op-attrs/tensor_dims.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

template <DataType DT>
struct CreateFilledAccessorW {
  GenericTensorAccessorW operator()(TensorShape const &shape,
                                    Allocator &allocator,
                                    DataTypeValue val) {
    using T = real_type_t<DT>;
    if (!val.template has<T>()) {
      PANIC("create_filed_accessor expected data type of "
            "shape and passed-in value to match");
    }

    auto unwrapped_value = val.get<T>();
    GenericTensorAccessorW dst_accessor = allocator.allocate_tensor(shape);
    Allocator cpu_allocator = create_local_cpu_memory_allocator();
    GenericTensorAccessorW src_accessor = cpu_allocator.allocate_tensor(shape);

    T *data_ptr = src_accessor.get<DT>();

    int volume =
        get_num_elements(dst_accessor.shape.dims).int_from_positive_int();
    for (size_t i = 0; i < volume; i++) {
      data_ptr[i] = unwrapped_value;
    }

    copy_accessor_data_to_l_from_r(dst_accessor, src_accessor);
    return dst_accessor;
  }
};

GenericTensorAccessorW create_constant_filled_accessor_w(
    TensorShape const &shape, Allocator &allocator, DataTypeValue val) {

  return DataTypeDispatch1<CreateFilledAccessorW>{}(
      shape.data_type, shape, allocator, val);
}

GenericTensorAccessorR create_constant_filled_accessor_r(
    TensorShape const &shape, Allocator &allocator, DataTypeValue val) {
  GenericTensorAccessorW w_accessor =
      create_constant_filled_accessor_w(shape, allocator, val);
  return read_only_accessor_from_write_accessor(w_accessor);
}

} // namespace FlexFlow
