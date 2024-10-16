#ifndef _FLEXFLOW_KERNELS_TEST_UTILS
#define _FLEXFLOW_KERNELS_TEST_UTILS

#include "kernels/datatype_dispatch.h"
#include "kernels/device.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/local_cuda_allocator.h"
#include "kernels/managed_ff_stream.h"
#include "kernels/managed_per_device_ff_handle.h"
#include "op-attrs/datatype.h"

namespace FlexFlow {

GenericTensorAccessorW create_random_filled_accessor_w(TensorShape const &shape,
                                                       Allocator &allocator);

GenericTensorAccessorR create_random_filled_accessor_r(TensorShape const &shape,
                                                       Allocator &allocator);

GenericTensorAccessorW create_zero_filled_accessor_w(TensorShape const &shape,
                                                     Allocator &allocator);

TensorShape
    make_tensor_shape_from_legion_dims(LegionOrdered<size_t> const &dims,
                                       DataType DT);

bool contains_non_zero(GenericTensorAccessorW const &accessor);

bool contains_non_zero(GenericTensorAccessorR const &accessor);

void fill_with_zeros(GenericTensorAccessorW const &accessor);

GenericTensorAccessorW
    create_cpu_compatible_accessor_w(GenericTensorAccessorW const &accessor,
                                     Allocator &allocator);

GenericTensorAccessorR
    create_cpu_compatible_accessor_r(GenericTensorAccessorR const &accessor,
                                     Allocator &allocator);

void print_accessor(GenericTensorAccessorR const &accessor);

void print_accessor(GenericTensorAccessorW const &accessor);

template <DataType DT>
struct CreateFilledAccessorW {
  GenericTensorAccessorW operator()(TensorShape const &shape,
                                    Allocator &allocator,
                                    real_type_t<DT> val) {
    using T = real_type_t<DT>;

    GenericTensorAccessorW dst_accessor = allocator.allocate_tensor(shape);

    Allocator cpu_allocator = create_local_cpu_memory_allocator();
    GenericTensorAccessorW src_accessor = cpu_allocator.allocate_tensor(shape);

    T *data_ptr = src_accessor.get<DT>();
    for (size_t i = 0; i < dst_accessor.shape.num_elements(); i++) {
      data_ptr[i] = val;
    }

    transfer_data_between_accessors(dst_accessor, src_accessor);
    return dst_accessor;
  }
};

template <typename T>
GenericTensorAccessorW create_filled_accessor_w(TensorShape const &shape,
                                                Allocator &allocator,
                                                T val) {
  return DataTypeDispatch1<CreateFilledAccessorW>{}(
      shape.data_type, shape, std::ref(allocator), val);
}

template <typename T>
GenericTensorAccessorR create_filled_accessor_r(TensorShape const &shape,
                                                Allocator &allocator,
                                                T val) {
  GenericTensorAccessorW w_accessor =
      create_filled_accessor_w(shape, allocator, val);
  return read_only_accessor_from_write_accessor(w_accessor);
}

template <DataType DT>
bool w_accessors_are_equal(GenericTensorAccessorW const &accessor_a,
                           GenericTensorAccessorW const &accessor_b) {
  if (accessor_a.shape.num_dims() != accessor_b.shape.num_dims()) {
    throw mk_runtime_error(
        "Comparing equivalence for two accessors of differing dimensions");
  }
  for (size_t i = 0; i < accessor_a.shape.num_dims(); i++) {
    if (accessor_a.shape[legion_dim_t(i)] !=
        accessor_b.shape[legion_dim_t(i)]) {
      throw mk_runtime_error(
          "Comparing equivalence for two accessors of differing shape");
    }
  }

  if (accessor_a.data_type != accessor_b.data_type) {
    return false;
  }

  Allocator cpu_allocator = create_local_cpu_memory_allocator();
  GenericTensorAccessorW cpu_accessor_a =
      create_cpu_compatible_accessor_w(accessor_a, cpu_allocator);
  GenericTensorAccessorW cpu_accessor_b =
      create_cpu_compatible_accessor_w(accessor_b, cpu_allocator);

  using T = real_type_t<DT>;
  T *a_data_ptr = cpu_accessor_a.get<DT>();
  T *b_data_ptr = cpu_accessor_b.get<DT>();

  for (size_t i = 0; i < accessor_a.shape.num_elements(); i++) {
    if (a_data_ptr[i] != b_data_ptr[i]) {
      print_accessor(cpu_accessor_a);
      print_accessor(cpu_accessor_b);
      return false;
    }
  }

  return true;
}

} // namespace FlexFlow

#endif
