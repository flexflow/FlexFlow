#include "kernels/tensor_accessor_unary_ops.h"
#include "kernels/datatype_dispatch.h"
#include "kernels/fill_tensor_accessor.h"
#include "kernels/map_tensor_accessors.h"
#include "op-attrs/datatype_value.h"
#include "op-attrs/ff_ordered/ff_ordered_concat.h"
#include "op-attrs/ff_ordered/ff_ordered_reversed.h"
#include "op-attrs/ff_ordered/ff_ordered_slice.h"
#include "op-attrs/tensor_dims.h"
#include "op-attrs/tensor_dims_coord.h"
#include <cmath>

namespace FlexFlow {

GenericTensorAccessorW tensor_accessor_exp(GenericTensorAccessorR const &input,
                                           Allocator &output_allocator) {
  ASSERT(input.shape.data_type == DataType::FLOAT);

  return map_tensor_accessor(
      input,
      [&](auto const &elem) { return std::exp(elem); },
      output_allocator);
}

void tensor_accessor_exp_to(GenericTensorAccessorR const &input,
                            GenericTensorAccessorW const &output) {
  ASSERT(input.shape.data_type == DataType::FLOAT);

  map_tensor_accessor_to(
      input, [](auto elem) { return std::exp(elem); }, output);
}

void tensor_accessor_exp_inplace(GenericTensorAccessorW const &input) {
  ASSERT(input.shape.data_type == DataType::FLOAT);

  map_tensor_accessor_inplace(input,
                              [&](auto &elem) { return std::exp(elem); });
}

GenericTensorAccessorW
    tensor_accessor_scale_by_constant(GenericTensorAccessorR const &t,
                                      float constant,
                                      Allocator &output_allocator) {
  ASSERT(t.shape.data_type == DataType::FLOAT);

  return map_tensor_accessor(
      t, [&](auto const &elem) { return elem * constant; }, output_allocator);
}

void tensor_accessor_scale_by_constant_inplace(GenericTensorAccessorW const &t,
                                               float constant) {
  ASSERT(t.shape.data_type == DataType::FLOAT);

  return map_tensor_accessor_inplace(
      t, [&](auto const &elem) { return elem * constant; });
}

template <typename T>
static T single_element_relu(T elem) {
  if (elem >= 0) {
    return elem;
  } else {
    return 0;
  }
}

void tensor_accessor_relu_to(GenericTensorAccessorR const &input,
                             GenericTensorAccessorW const &output) {
  map_tensor_accessor_to(
      input, [](auto elem) { return single_element_relu(elem); }, output);
}

GenericTensorAccessorW tensor_accessor_relu(GenericTensorAccessorR const &input,
                                            Allocator &output_allocator) {
  return map_tensor_accessor(
      input,
      [](auto elem) { return single_element_relu(elem); },
      output_allocator);
}

template <DataType DT>
struct CPUTensorAccessorBroadcast {
  void operator()(GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output) {

    for (TensorDimsCoord const &output_coord :
         get_tensor_dims_coord_set(output.shape.dims)) {
      TensorDimsCoord input_coord = get_broadcast_src_coord(
          /*input_dims=*/input.shape.dims,
          /*output_dims=*/output.shape.dims,
          /*dst_coord=*/output_coord);

      output.at<DT>(output_coord) = input.at<DT>(input_coord);
    }
  }
};

void tensor_accessor_broadcast_to(GenericTensorAccessorR const &input,
                                  TensorDims const &output_dims,
                                  GenericTensorAccessorW const &output) {
  ASSERT(tensor_dims_is_broadcastable_to(input.shape.dims, output_dims));

  TensorShape output_shape = TensorShape{output_dims, input.shape.data_type};
  ASSERT(get_tensor_shape_for_accessor_w(output) == output_shape);

  Allocator cpu_allocator = create_local_cpu_memory_allocator();
  GenericTensorAccessorR input_cpu =
      copy_tensor_accessor_r_to_cpu_if_necessary(input, cpu_allocator);

  GenericTensorAccessorW output_cpu =
      cpu_allocator.allocate_tensor(output_shape);

  DataTypeDispatch1<CPUTensorAccessorBroadcast>{}(
      input.shape.data_type, input_cpu, output_cpu);

  copy_accessor_data_to_l_from_r(output, output_cpu);
}

GenericTensorAccessorW
    tensor_accessor_broadcast(GenericTensorAccessorR const &input,
                              TensorDims const &output_dims,
                              Allocator &output_allocator) {

  TensorShape output_shape = TensorShape{output_dims, input.shape.data_type};

  GenericTensorAccessorW output =
      output_allocator.allocate_tensor(output_shape);

  tensor_accessor_broadcast_to(input, output_dims, output);

  return output;
}

template <DataType DT>
struct CPUTensorAccessorTranspose {
  void operator()(GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output,
                  TensorDimPermutation const &dim_permutation) {

    for (TensorDimsCoord const &input_coord :
         get_tensor_dims_coord_set(input.shape.dims)) {

      TensorDimsCoord output_coord =
          permute_tensor_dims_coord(dim_permutation, input_coord);

      output.at<DT>(output_coord) = input.at<DT>(input_coord);
    }
  }
};

static TensorShape
    get_transpose_output_shape(TensorShape const &input_shape,
                               TensorDimPermutation const &dim_permutation) {
  return permute_tensor_shape(dim_permutation, input_shape);
}

void tensor_accessor_transpose_to(GenericTensorAccessorR const &input,
                                  TensorDimPermutation const &dim_permutation,
                                  GenericTensorAccessorW const &output) {
  TensorShape output_shape = get_transpose_output_shape(
      get_tensor_shape_for_accessor_r(input), dim_permutation);
  ASSERT(get_tensor_shape_for_accessor_w(output) == output_shape);

  Allocator cpu_allocator = create_local_cpu_memory_allocator();
  GenericTensorAccessorR input_cpu =
      copy_tensor_accessor_r_to_cpu_if_necessary(input, cpu_allocator);

  GenericTensorAccessorW output_cpu =
      cpu_allocator.allocate_tensor(output_shape);

  DataTypeDispatch1<CPUTensorAccessorTranspose>{}(
      input.shape.data_type, input_cpu, output_cpu, dim_permutation);

  copy_accessor_data_to_l_from_r(output, output_cpu);
}

GenericTensorAccessorW
    tensor_accessor_transpose(GenericTensorAccessorR const &input,
                              TensorDimPermutation const &dim_permutation,
                              Allocator &output_allocator) {

  TensorShape output_shape = get_transpose_output_shape(
      get_tensor_shape_for_accessor_r(input), dim_permutation);

  GenericTensorAccessorW output =
      output_allocator.allocate_tensor(output_shape);

  tensor_accessor_transpose_to(input, dim_permutation, output);

  return output;
}

static TensorDimPermutation get_2d_transpose_permutation() {
  ff_dim_t d0 = ff_dim_t{0_n};
  ff_dim_t d1 = ff_dim_t{1_n};

  TensorDimPermutation permutation = TensorDimPermutation{
      bidict<ff_dim_t, ff_dim_t>{
          {d0, d1},
          {d1, d0},
      },
  };

  return permutation;
}

GenericTensorAccessorW
    tensor_accessor_transpose_2d(GenericTensorAccessorR const &input,
                                 Allocator &output_allocator) {
  ASSERT(get_num_dims(input.shape.dims) == 2);

  TensorDimPermutation permutation = get_2d_transpose_permutation();

  return tensor_accessor_transpose(input, permutation, output_allocator);
}

void tensor_accessor_transpose_2d_to(GenericTensorAccessorR const &input,
                                     GenericTensorAccessorW const &output) {
  ASSERT(get_num_dims(input.shape.dims) == 2);

  TensorDimPermutation permutation = get_2d_transpose_permutation();

  return tensor_accessor_transpose_to(input, permutation, output);
}

static TensorDimPermutation get_batch_transpose_permutation() {
  ff_dim_t d0 = ff_dim_t{0_n};
  ff_dim_t d1 = ff_dim_t{1_n};
  ff_dim_t d2 = ff_dim_t{2_n};

  TensorDimPermutation permutation = TensorDimPermutation{
      bidict<ff_dim_t, ff_dim_t>{
          {d0, d0},
          {d1, d2},
          {d2, d1},
      },
  };

  return permutation;
}

void tensor_accessor_batch_transpose_to(GenericTensorAccessorR const &input,
                                        GenericTensorAccessorW const &output) {
  ASSERT(get_num_dims(input.shape.dims) == 3);

  TensorDimPermutation permutation = get_batch_transpose_permutation();

  return tensor_accessor_transpose_to(input, permutation, output);
}

GenericTensorAccessorW
    tensor_accessor_batch_transpose(GenericTensorAccessorR const &input,
                                    Allocator &output_allocator) {
  ASSERT(get_num_dims(input.shape.dims) == 3);

  TensorDimPermutation permutation = get_batch_transpose_permutation();

  return tensor_accessor_transpose(input, permutation, output_allocator);
}

template <DataType DT>
struct CPUTensorAccessorReduce {
  void operator()(GenericTensorAccessorR const &input,
                  ff_dim_t reduction_dim,
                  GenericTensorAccessorW const &output) {
    fill_with_zeros(output);

    for (TensorDimsCoord const &input_coord :
         get_tensor_dims_coord_set(input.shape.dims)) {
      TensorDimsCoord output_coord = tensor_dims_coord_drop_dims(
          input_coord, [&](ff_dim_t input_coord_dim) {
            return input_coord_dim == reduction_dim;
          });

      output.at<DT>(output_coord) += input.at<DT>(input_coord);
    }
  }
};

static TensorShape get_reduce_output_shape(TensorShape const &input_shape,
                                           ff_dim_t reduction_dim) {
  ASSERT(tensor_dims_has_dim(input_shape.dims, reduction_dim),
         input_shape.dims,
         reduction_dim);

  return TensorShape{
      TensorDims{
          ff_ordered_concat(
              ff_ordered_slice(
                  input_shape.dims.ff_ordered, ff_dim_t{0_n}, reduction_dim),
              ff_ordered_slice(input_shape.dims.ff_ordered,
                               ff_dim_t{reduction_dim.value + 1_n},
                               std::nullopt)),
      },
      input_shape.data_type,
  };
}

void tensor_accessor_reduce_to(GenericTensorAccessorR const &input,
                               ff_dim_t reduction_dim,
                               GenericTensorAccessorW const &output) {

  TensorShape output_shape = get_reduce_output_shape(
      get_tensor_shape_for_accessor_r(input), reduction_dim);
  ASSERT(get_tensor_shape_for_accessor_r(output) == output_shape);

  Allocator cpu_allocator = create_local_cpu_memory_allocator();
  GenericTensorAccessorR input_cpu =
      copy_tensor_accessor_r_to_cpu_if_necessary(input, cpu_allocator);

  GenericTensorAccessorW output_cpu =
      cpu_allocator.allocate_tensor(output_shape);

  DataTypeDispatch1<CPUTensorAccessorReduce>{}(
      input.shape.data_type, input_cpu, reduction_dim, output_cpu);

  copy_accessor_data_to_l_from_r(output, output_cpu);
}

GenericTensorAccessorW
    tensor_accessor_reduce(GenericTensorAccessorR const &input,
                           ff_dim_t reduction_dim,
                           Allocator &output_allocator) {

  TensorShape output_shape = get_reduce_output_shape(
      get_tensor_shape_for_accessor_r(input), reduction_dim);

  GenericTensorAccessorW output =
      output_allocator.allocate_tensor(output_shape);

  tensor_accessor_reduce_to(input, reduction_dim, output);

  return output;
}

} // namespace FlexFlow
