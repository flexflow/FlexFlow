#include "test_utils.h"
#include "op-attrs/tensor_shape.h"
#include <random>

namespace FlexFlow {

GenericTensorAccessorW create_zero_filled_accessor_w(TensorShape const &shape,
                                                     Allocator &allocator) {
  GenericTensorAccessorW result_accessor = allocator.allocate_tensor(shape);
  fill_with_zeros(result_accessor);
  return result_accessor;
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
struct CreateRandomFilledAccessorW {
  GenericTensorAccessorW operator()(TensorShape const &shape,
                                    Allocator &allocator) {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();
    GenericTensorAccessorW src_accessor = cpu_allocator.allocate_tensor(shape);

    using T = real_type_t<DT>;
    T *data_ptr = src_accessor.get<DT>();

    std::random_device rd;
    std::mt19937 gen(rd());
    size_t num_elements = get_num_elements(shape);
    if constexpr (std::is_same<T, bool>::value) {
      std::bernoulli_distribution dist(0.5);
      for (size_t i = 0; i < num_elements; i++) {
        data_ptr[i] = dist(gen);
      }
    } else if constexpr (std::is_floating_point<T>::value) {
      std::uniform_real_distribution<T> dist(-1.0, 1.0);
      for (size_t i = 0; i < num_elements; i++) {
        data_ptr[i] = dist(gen);
      }
    } else if constexpr (std::is_integral<T>::value) {
      std::uniform_int_distribution<T> dist(0, 100);
      for (size_t i = 0; i < num_elements; i++) {
        data_ptr[i] = dist(gen);
      }
    }

    GenericTensorAccessorW dst_accessor = allocator.allocate_tensor(shape);
    transfer_data_between_accessors(dst_accessor, src_accessor);

    return dst_accessor;
  }
};

GenericTensorAccessorW create_random_filled_accessor_w(TensorShape const &shape,
                                                       Allocator &allocator) {
  return DataTypeDispatch1<CreateRandomFilledAccessorW>{}(
      shape.data_type, shape, std::ref(allocator));
}

GenericTensorAccessorR create_random_filled_accessor_r(TensorShape const &shape,
                                                       Allocator &allocator) {
  GenericTensorAccessorW accessor =
      create_random_filled_accessor_w(shape, allocator);

  return read_only_accessor_from_write_accessor(accessor);
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

template <DataType DT>
struct CPUAccessorRContainsNonZero {
  bool operator()(GenericTensorAccessorR const &accessor) {
    using T = real_type_t<DT>;

    T const *data_ptr = accessor.get<DT>();

    for (size_t i = 0; i < accessor.shape.num_elements(); i++) {
      if (data_ptr[i] != 0) {
        return true;
      }
    }

    return false;
  }
};

bool contains_non_zero(GenericTensorAccessorR const &accessor) {
  Allocator cpu_allocator = create_local_cpu_memory_allocator();
  GenericTensorAccessorR cpu_accessor =
      create_cpu_compatible_accessor_r(accessor, cpu_allocator);
  return DataTypeDispatch1<CPUAccessorRContainsNonZero>{}(
      cpu_accessor.data_type, cpu_accessor);
}

bool contains_non_zero(GenericTensorAccessorW const &accessor) {
  GenericTensorAccessorR r_accessor =
      read_only_accessor_from_write_accessor(accessor);
  return contains_non_zero(r_accessor);
}

GenericTensorAccessorR
    create_cpu_compatible_accessor_r(GenericTensorAccessorR const &accessor,
                                     Allocator &cpu_allocator) {
  GenericTensorAccessorR cpu_accessor = accessor;
  if (accessor.device_type == DeviceType::GPU) {
    cpu_accessor = copy_tensor_accessor_r(accessor, cpu_allocator);
  }
  return cpu_accessor;
}

GenericTensorAccessorW
    create_cpu_compatible_accessor_w(GenericTensorAccessorW const &accessor,
                                     Allocator &cpu_allocator) {
  GenericTensorAccessorW cpu_accessor = accessor;
  if (accessor.device_type == DeviceType::GPU) {
    cpu_accessor = copy_tensor_accessor_w(accessor, cpu_allocator);
  }
  return cpu_accessor;
}

template <DataType DT>
struct PrintCPUAccessorR {
  void operator()(GenericTensorAccessorR const &accessor) {
    using T = real_type_t<DT>;

    T const *data_ptr = accessor.get<DT>();
    for (size_t i = 0; i < accessor.shape.num_elements(); i++) {
      std::cout << data_ptr[i] << " ";
    }
    std::cout << "\n";
  }
};

void print_accessor(GenericTensorAccessorR const &accessor) {
  Allocator cpu_allocator = create_local_cpu_memory_allocator();
  GenericTensorAccessorR cpu_accessor =
      create_cpu_compatible_accessor_r(accessor, cpu_allocator);
  DataTypeDispatch1<PrintCPUAccessorR>{}(accessor.data_type, accessor);
}

void print_accessor(GenericTensorAccessorW const &accessor) {
  GenericTensorAccessorR r_accessor =
      read_only_accessor_from_write_accessor(accessor);
  print_accessor(r_accessor);
}

} // namespace FlexFlow
