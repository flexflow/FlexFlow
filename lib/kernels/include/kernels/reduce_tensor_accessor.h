#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_REDUCE_TENSOR_ACCESSOR_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_REDUCE_TENSOR_ACCESSOR_H

#include "kernels/accessor.h"
#include "kernels/accessors_are_approximately_equal.h"
#include "kernels/allocation.h"
#include "kernels/copy_tensor_accessor.h"
#include "kernels/datatype_dispatch.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/values_are_approximately_equal.h"
#include "op-attrs/tensor_dims_coord.h"
#include "op-attrs/tensor_shape.h"
#include "utils/containers/contains.h"
#include "utils/containers/foldl1.h"
#include "utils/containers/foldr1.h"
#include "utils/containers/group_by.h"
#include "utils/containers/sorted.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

template <DataType DT>
struct CPUReduceTensorAccessorInDims {
  template <typename F>
  void operator()(GenericTensorAccessorR const &input,
                  GenericTensorAccessorW &output,
                  std::set<ff_dim_t> const &dims_to_reduce,
                  F &&f) {
    using T = real_type_t<DT>;

    ASSERT(input.device_type == DeviceType::CPU);
    ASSERT(output.device_type == DeviceType::CPU);

    auto should_drop_dim = [&](ff_dim_t dim) -> bool {
      return contains(dims_to_reduce, dim);
    };

    OneToMany<TensorDimsCoord, TensorDimsCoord> output_coord_from_input_coord =
        group_by(get_tensor_dims_coord_set(input.shape.dims),
                 [&](TensorDimsCoord const &input_coord) {
                   return tensor_dims_coord_drop_dims(input_coord,
                                                      should_drop_dim);
                 });

    for (auto const &[output_coord, input_coords] :
         output_coord_from_input_coord.l_to_r()) {
      std::vector<T> input_values = transform(
          sorted(input_coords), [&](TensorDimsCoord const &input_coord) -> T {
            return input.at<DT>(input_coord);
          });

      T result = foldl1(input_values, f);

#ifndef NDEBUG
      {
        T flipped_result =
            foldr1(input_values, [&](T const &accum, T const &elem) {
              return f(elem, accum);
            });

        ASSERT(values_are_approximately_equal(result, flipped_result));
      }
#endif

      output.at<DT>(output_coord) = result;
    }
  }
};

template <typename F>
GenericTensorAccessorW
    reduce_tensor_accessor_in_dims(GenericTensorAccessorR const &input,
                                   std::set<ff_dim_t> const &dims,
                                   Allocator &output_allocator,
                                   F &&f) {

  Allocator cpu_allocator = create_local_cpu_memory_allocator();
  GenericTensorAccessorR input_cpu =
      copy_tensor_accessor_r_to_cpu_if_necessary(input, cpu_allocator);

  auto should_drop_dim = [&](ff_dim_t dim) -> bool {
    return contains(dims, dim);
  };

  TensorShape reduced_shape =
      tensor_shape_drop_dims(input.shape, should_drop_dim);
  GenericTensorAccessorW output_cpu =
      cpu_allocator.allocate_tensor(reduced_shape);

  DataTypeDispatch1<CPUReduceTensorAccessorInDims>{}(
      input_cpu.shape.data_type, input_cpu, output_cpu, dims, f);

  return copy_tensor_accessor_w(output_cpu, output_allocator);
}

template <DataType DT, typename F>
real_type_t<DT>
    reduce_tensor_accessor_in_all_dims(GenericTensorAccessorR const &input,
                                       F &&f) {
  Allocator cpu_allocator = create_local_cpu_memory_allocator();

  std::set<ff_dim_t> input_dims = get_ff_dim_t_set(input.shape.dims);
  GenericTensorAccessorW reduced =
      reduce_tensor_accessor_in_dims(input, input_dims, cpu_allocator, f);

  return accessor_get_only_value<DT>(reduced);
}

} // namespace FlexFlow

#endif
