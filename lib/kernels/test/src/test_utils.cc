#include "test_utils.h"

namespace FlexFlow {

bool device_on_cpu(DeviceType device_type) {
  return device_type == DeviceType::CPU;
}

bool device_on_gpu(DeviceType device_type) {
  return device_type == DeviceType::GPU;
}

TensorShape
    make_tensor_shape_from_legion_dims(LegionOrdered<size_t> const &dims,
                                       DataType DT) {
  return TensorShape{
      TensorDims{
          ff_ordered_from_legion_ordered(dims),
      },
      DT,
  };
}

template <DataType DT>
struct CopyTensorAccessorW {
  GenericTensorAccessorW operator()(GenericTensorAccessorW const &src_accessor,
                                    Allocator &allocator) {
    TensorShape shape =
        get_tensor_shape(src_accessor.shape, src_accessor.data_type);
    GenericTensorAccessorW copied_tensor = allocator.allocate_tensor(shape);

    transfer_memory(
        copied_tensor, src_accessor.get<DT>(), src_accessor.device_type);

    return copied_tensor;
  }
};

GenericTensorAccessorW
    copy_tensor_accessor_w(GenericTensorAccessorW const &src_accessor,
                           Allocator &allocator) {
  return DataTypeDispatch1<CopyTensorAccessorW>{}(
      src_accessor.data_type, src_accessor, std::ref(allocator));
}

template <DataType DT>
struct CopyTensorAccessorR {
  GenericTensorAccessorR operator()(GenericTensorAccessorR const &src_accessor,
                                    Allocator &allocator) {
    TensorShape shape =
        get_tensor_shape(src_accessor.shape, src_accessor.data_type);
    GenericTensorAccessorW copied_tensor = allocator.allocate_tensor(shape);

    transfer_memory(
        copied_tensor, src_accessor.get<DT>(), src_accessor.device_type);

    return read_only_accessor_from_write_accessor(copied_tensor);
  }
};

GenericTensorAccessorR
    copy_tensor_accessor_r(GenericTensorAccessorR const &src_accessor,
                           Allocator &allocator) {
  return DataTypeDispatch1<CopyTensorAccessorR>{}(
      src_accessor.data_type, src_accessor, std::ref(allocator));
}

template <DataType DT>
struct FillWithZeros {
  void operator()(GenericTensorAccessorW const &accessor) {
    using T = real_type_t<DT>;

    if (accessor.device_type == DeviceType::CPU) {
      memset(accessor.ptr, 0, accessor.shape.get_volume() * sizeof(T));
    } else {
      checkCUDA(
          cudaMemset(accessor.ptr, 0, accessor.shape.get_volume() * sizeof(T)));
    }
  }
};

void fill_with_zeros(GenericTensorAccessorW const &accessor) {
  DataTypeDispatch1<FillWithZeros>{}(accessor.data_type, accessor);
}

} // namespace FlexFlow
