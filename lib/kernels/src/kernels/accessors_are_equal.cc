#include "kernels/accessors_are_equal.h"
#include "kernels/copy_tensor_accessor.h"
#include "kernels/datatype_dispatch.h"
#include "kernels/local_cpu_allocator.h"
#include "op-attrs/tensor_dims.h"

namespace FlexFlow {

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

} // namespace FlexFlow
