#include "internal/test_utils.h"
#include "kernels/fill_tensor_accessor.h"
#include "kernels/map_tensor_accessors.h"
#include "kernels/tensor_accessor_binary_ops.h"
#include "kernels/tensor_accessor_reductions.h"
#include "op-attrs/tensor_shape.h"
#include "utils/containers/require_all_same1.h"
#include "utils/exception.h"
#include "utils/join_strings.h"
#include <random>

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

template <DataType DT>
struct CreateRandomFilledAccessorW {
  GenericTensorAccessorW operator()(TensorShape const &shape,
                                    Allocator &allocator) {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();
    GenericTensorAccessorW src_accessor = cpu_allocator.allocate_tensor(shape);

    using T = real_type_t<DT>;
    T *data_ptr = src_accessor.get<DT>();

    std::mt19937 gen(0); // seed with 0 so tests are deterministic
    size_t num_elements = get_num_elements(shape.dims).int_from_positive_int();
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
      std::uniform_int_distribution<T> dist(0, 99);
      for (size_t i = 0; i < num_elements; i++) {
        data_ptr[i] = dist(gen);
      }
    }

    GenericTensorAccessorW dst_accessor = allocator.allocate_tensor(shape);
    copy_accessor_data_to_l_from_r(dst_accessor, src_accessor);

    return dst_accessor;
  }
};

GenericTensorAccessorW create_random_filled_accessor_w(TensorShape const &shape,
                                                       Allocator &allocator) {
  return DataTypeDispatch1<CreateRandomFilledAccessorW>{}(
      shape.data_type, shape, allocator);
}

GenericTensorAccessorR create_random_filled_accessor_r(TensorShape const &shape,
                                                       Allocator &allocator) {
  GenericTensorAccessorW accessor =
      create_random_filled_accessor_w(shape, allocator);

  return read_only_accessor_from_write_accessor(accessor);
}

template <DataType DT>
struct CPUAccessorRContainsNonZero {
  bool operator()(GenericTensorAccessorR const &accessor) {
    using T = real_type_t<DT>;

    T const *data_ptr = accessor.get<DT>();

    int volume = get_num_elements(accessor.shape.dims).int_from_positive_int();
    for (size_t i = 0; i < volume; i++) {
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
      copy_tensor_accessor_r_to_cpu_if_necessary(accessor, cpu_allocator);
  return DataTypeDispatch1<CPUAccessorRContainsNonZero>{}(
      cpu_accessor.shape.data_type, cpu_accessor);
}

template <DataType DT>
struct AccessorsAreEqual {
  bool operator()(GenericTensorAccessorR const &accessor_a,
                  GenericTensorAccessorR const &accessor_b) {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();
    GenericTensorAccessorR cpu_accessor_a =
        copy_tensor_accessor_r_to_cpu_if_necessary(accessor_a, cpu_allocator);
    GenericTensorAccessorR cpu_accessor_b =
        copy_tensor_accessor_r_to_cpu_if_necessary(accessor_b, cpu_allocator);

    using T = real_type_t<DT>;
    T const *a_data_ptr = cpu_accessor_a.get<DT>();
    T const *b_data_ptr = cpu_accessor_b.get<DT>();

    int volume =
        get_num_elements(accessor_a.shape.dims).int_from_positive_int();
    for (size_t i = 0; i < volume; i++) {
      if (a_data_ptr[i] != b_data_ptr[i]) {
        return false;
      }
    }

    return true;
  }
};

bool accessors_are_equal(GenericTensorAccessorR const &accessor_a,
                         GenericTensorAccessorR const &accessor_b) {
  ASSERT(accessor_a.shape == accessor_b.shape,
         "accessors_are_equal expects accessors to have the same shape");

  return DataTypeDispatch1<AccessorsAreEqual>{}(
      accessor_a.shape.data_type, accessor_a, accessor_b);
}

bool accessors_within_epsilon(GenericTensorAccessorR const &accessor_a,
                              GenericTensorAccessorR const &accessor_b,
                              float epsilon) {
  ASSERT(accessor_a.shape == accessor_b.shape,
         "accessors_are_equal expects accessors to have the same shape");

  Allocator cpu_allocator = create_local_cpu_memory_allocator();
  GenericTensorAccessorW diff = tensor_accessor_elementwise_subtract(
      accessor_a, accessor_b, cpu_allocator);
  GenericTensorAccessorW within_epsilon = map_tensor_accessor(
      diff, [&](float x) { return std::abs(x) < epsilon; }, cpu_allocator);
  return tensor_accessor_all(within_epsilon);
}

template <DataType DT>
struct CreateFilledAccessorW {
  GenericTensorAccessorW operator()(TensorShape const &shape,
                                    Allocator &allocator,
                                    DataTypeValue val) {
    using T = real_type_t<DT>;
    if (!val.template has<T>()) {
      throw mk_runtime_error("create_filed_accessor expected data type of "
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

GenericTensorAccessorW create_filled_accessor_w(TensorShape const &shape,
                                                Allocator &allocator,
                                                DataTypeValue val) {

  return DataTypeDispatch1<CreateFilledAccessorW>{}(
      shape.data_type, shape, allocator, val);
}

GenericTensorAccessorR create_filled_accessor_r(TensorShape const &shape,
                                                Allocator &allocator,
                                                DataTypeValue val) {
  GenericTensorAccessorW w_accessor =
      create_filled_accessor_w(shape, allocator, val);
  return read_only_accessor_from_write_accessor(w_accessor);
}
} // namespace FlexFlow
