#include "kernels/create_random_filled_accessor.h"
#include "kernels/accessor.h"
#include "kernels/datatype_dispatch.h"
#include "kernels/local_cpu_allocator.h"
#include "op-attrs/tensor_dims.h"
#include <random>

namespace FlexFlow {

template <DataType DT>
struct CreateRandomFilledAccessorW {
  GenericTensorAccessorW operator()(TensorShape const &shape,
                                    Allocator &allocator,
                                    std::mt19937 &gen) {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();
    GenericTensorAccessorW src_accessor = cpu_allocator.allocate_tensor(shape);

    using T = real_type_t<DT>;
    T *data_ptr = src_accessor.get<DT>();

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
                                                       Allocator &allocator,
                                                       int seed) {
  std::mt19937 gen(seed);

  return create_random_filled_accessor_w_with_gen(shape, allocator, gen);
}

GenericTensorAccessorR create_random_filled_accessor_r(TensorShape const &shape,
                                                       Allocator &allocator,
                                                       int seed) {
  GenericTensorAccessorW accessor =
      create_random_filled_accessor_w(shape, allocator, seed);

  return read_only_accessor_from_write_accessor(accessor);
}

GenericTensorAccessorW create_random_filled_accessor_w_with_gen(
    TensorShape const &shape, Allocator &allocator, std::mt19937 &gen) {
  return DataTypeDispatch1<CreateRandomFilledAccessorW>{}(
      shape.data_type, shape, allocator, gen);
}

GenericTensorAccessorR create_random_filled_accessor_r_with_gen(
    TensorShape const &shape, Allocator &allocator, std::mt19937 &gen) {
  GenericTensorAccessorW accessor =
      create_random_filled_accessor_w_with_gen(shape, allocator, gen);

  return read_only_accessor_from_write_accessor(accessor);
}

} // namespace FlexFlow
