#include "kernels/tensor_accessor_binary_ops.h"
#include "kernels/map_tensor_accessors.h"
#include "op-attrs/ff_ordered/ff_ordered_from_map.h"
#include "op-attrs/ff_ordered/map_from_ff_ordered.h"
#include "op-attrs/tensor_dims_coord.h"
#include "op-attrs/tensor_shape.h"
#include "utils/containers/generate_map.h"

namespace FlexFlow {

GenericTensorAccessorW
    tensor_accessor_elementwise_add(GenericTensorAccessorR const &lhs,
                                    GenericTensorAccessorR const &rhs,
                                    Allocator &output_allocator) {
  return map_tensor_accessors2(
      lhs,
      rhs,
      require_same(lhs.shape.data_type, rhs.shape.data_type),
      [](auto const &l, auto const &r) { return l + r; },
      output_allocator);
}

void tensor_accessor_elementwise_add_to(GenericTensorAccessorR const &lhs,
                                        GenericTensorAccessorR const &rhs,
                                        GenericTensorAccessorW const &output) {
  map_tensor_accessors2_to(
      lhs,
      rhs,
      require_same(lhs.shape.data_type, rhs.shape.data_type),
      [](auto const &l, auto const &r) { return l + r; },
      output);
}

GenericTensorAccessorW
    tensor_accessor_elementwise_subtract(GenericTensorAccessorR const &lhs,
                                         GenericTensorAccessorR const &rhs,
                                         Allocator &output_allocator) {
  return map_tensor_accessors2(
      lhs,
      rhs,
      require_same(lhs.shape.data_type, rhs.shape.data_type),
      [](auto const &l, auto const &r) { return l - r; },
      output_allocator);
}

void tensor_accessor_elementwise_subtract_to(
    GenericTensorAccessorR const &lhs,
    GenericTensorAccessorR const &rhs,
    GenericTensorAccessorW const &output) {
  map_tensor_accessors2_to(
      lhs,
      rhs,
      require_same(lhs.shape.data_type, rhs.shape.data_type),
      [](auto const &l, auto const &r) { return l - r; },
      output);
}

GenericTensorAccessorW
    tensor_accessor_elementwise_multiply(GenericTensorAccessorR const &lhs,
                                         GenericTensorAccessorR const &rhs,
                                         Allocator &output_allocator) {
  return map_tensor_accessors2(
      lhs,
      rhs,
      require_same(lhs.shape.data_type, rhs.shape.data_type),
      [](auto const &l, auto const &r) { return l * r; },
      output_allocator);
}

void tensor_accessor_elementwise_multiply_to(
    GenericTensorAccessorR const &lhs,
    GenericTensorAccessorR const &rhs,
    GenericTensorAccessorW const &output) {
  map_tensor_accessors2_to(
      lhs,
      rhs,
      require_same(lhs.shape.data_type, rhs.shape.data_type),
      [](auto const &l, auto const &r) { return l * r; },
      output);
}

GenericTensorAccessorW
    tensor_accessor_elementwise_divide(GenericTensorAccessorR const &lhs,
                                       GenericTensorAccessorR const &rhs,
                                       Allocator &output_allocator) {
  return map_tensor_accessors2(
      lhs,
      rhs,
      require_same(lhs.shape.data_type, rhs.shape.data_type),
      [](auto const &l, auto const &r) { return l / r; },
      output_allocator);
}

void tensor_accessor_elementwise_divide_to(
    GenericTensorAccessorR const &lhs,
    GenericTensorAccessorR const &rhs,
    GenericTensorAccessorW const &output) {
  map_tensor_accessors2_to(
      lhs,
      rhs,
      require_same(lhs.shape.data_type, rhs.shape.data_type),
      [](auto const &l, auto const &r) { return l / r; },
      output);
}

static TensorShape get_matmul_output_shape(TensorShape const &lhs,
                                           TensorShape const &rhs) {
  ASSERT(get_num_dims(lhs.dims) == 2);
  ASSERT(get_num_dims(rhs.dims) == 2);
  ASSERT(lhs.data_type == DataType::FLOAT);
  ASSERT(rhs.data_type == DataType::FLOAT);
  ASSERT(dim_at_idx(lhs.dims, relative_ff_dim_t{1}) ==
         dim_at_idx(rhs.dims, relative_ff_dim_t{0}));

  return TensorShape{
      TensorDims{FFOrdered{
          dim_at_idx(lhs.dims, relative_ff_dim_t{0}),
          dim_at_idx(rhs.dims, relative_ff_dim_t{1}),
      }},
      DataType::FLOAT,
  };
}

GenericTensorAccessorW
    tensor_accessor_elementwise_max(GenericTensorAccessorR const &lhs,
                                    GenericTensorAccessorR const &rhs,
                                    Allocator &output_allocator) {
  return map_tensor_accessors2(
      lhs,
      rhs,
      require_same(lhs.shape.data_type, rhs.shape.data_type),
      [](auto const &l, auto const &r) { return (l < r) ? r : l; },
      output_allocator);
}

void tensor_accessor_elementwise_max_to(GenericTensorAccessorR const &lhs,
                                        GenericTensorAccessorR const &rhs,
                                        GenericTensorAccessorW const &output) {
  map_tensor_accessors2_to(
      lhs,
      rhs,
      require_same(lhs.shape.data_type, rhs.shape.data_type),
      [](auto const &l, auto const &r) { return (l < r) ? r : l; },
      output);
}

GenericTensorAccessorW
    tensor_accessor_elementwise_min(GenericTensorAccessorR const &lhs,
                                    GenericTensorAccessorR const &rhs,
                                    Allocator &output_allocator) {
  return map_tensor_accessors2(
      lhs,
      rhs,
      require_same(lhs.shape.data_type, rhs.shape.data_type),
      [](auto const &l, auto const &r) { return (l < r) ? l : r; },
      output_allocator);
}

void tensor_accessor_elementwise_min_to(GenericTensorAccessorR const &lhs,
                                        GenericTensorAccessorR const &rhs,
                                        GenericTensorAccessorW const &output) {
  map_tensor_accessors2_to(
      lhs,
      rhs,
      require_same(lhs.shape.data_type, rhs.shape.data_type),
      [](auto const &l, auto const &r) { return (l < r) ? l : r; },
      output);
}

GenericTensorAccessorW tensor_accessor_matmul(GenericTensorAccessorR const &lhs,
                                              GenericTensorAccessorR const &rhs,
                                              Allocator &output_allocator) {
  TensorShape output_shape =
      get_matmul_output_shape(get_tensor_shape_for_accessor_r(lhs),
                              get_tensor_shape_for_accessor_r(rhs));

  GenericTensorAccessorW output =
      output_allocator.allocate_tensor(output_shape);

  tensor_accessor_matmul_to(lhs, rhs, output);

  return output;
}

void tensor_accessor_matmul_to(GenericTensorAccessorR const &lhs,
                               GenericTensorAccessorR const &rhs,
                               GenericTensorAccessorW const &output) {
  TensorShape output_shape =
      get_matmul_output_shape(get_tensor_shape_for_accessor_r(lhs),
                              get_tensor_shape_for_accessor_r(rhs));

  Allocator cpu_allocator = create_local_cpu_memory_allocator();
  GenericTensorAccessorR lhs_cpu =
      copy_tensor_accessor_r_to_cpu_if_necessary(lhs, cpu_allocator);
  GenericTensorAccessorR rhs_cpu =
      copy_tensor_accessor_r_to_cpu_if_necessary(rhs, cpu_allocator);
  GenericTensorAccessorW output_cpu =
      cpu_allocator.allocate_tensor(output_shape);

  for (nonnegative_int i :
       nonnegative_range(dim_at_idx(lhs.shape.dims, ff_dim_t{0_n}))) {
    for (nonnegative_int j :
         nonnegative_range(dim_at_idx(rhs.shape.dims, ff_dim_t{1_n}))) {
      float accum = 0.0f;
      for (nonnegative_int k :
           nonnegative_range(dim_at_idx(lhs.shape.dims, ff_dim_t{1_n}))) {
        accum += lhs_cpu.at<DataType::FLOAT>(TensorDimsCoord{FFOrdered{i, k}}) *
                 rhs_cpu.at<DataType::FLOAT>(TensorDimsCoord{FFOrdered{k, j}});
      }
      output_cpu.at<DataType::FLOAT>(TensorDimsCoord{FFOrdered{i, j}}) = accum;
    }
  }

  return copy_accessor_data_to_l_from_r(output, output_cpu);
}

static TensorShape get_batch_matmul_output_shape(TensorShape const &lhs,
                                                 TensorShape const &rhs) {
  ASSERT(get_num_dims(lhs.dims) == 3);
  ASSERT(get_num_dims(rhs.dims) == 3);
  ASSERT(lhs.data_type == DataType::FLOAT);
  ASSERT(rhs.data_type == DataType::FLOAT);

  positive_int batch_size = require_same(dim_at_idx(lhs.dims, ff_dim_t{0_n}),
                                         dim_at_idx(rhs.dims, ff_dim_t{0_n}));

  positive_int lhs_rows = dim_at_idx(lhs.dims, ff_dim_t{1_n});

  ASSERT(dim_at_idx(lhs.dims, ff_dim_t{2_n}) ==
         dim_at_idx(rhs.dims, ff_dim_t{1_n}));

  positive_int rhs_columns = dim_at_idx(rhs.dims, ff_dim_t{2_n});

  return TensorShape{
      TensorDims{FFOrdered{batch_size, lhs_rows, rhs_columns}},
      DataType::FLOAT,
  };
}

GenericTensorAccessorW
    tensor_accessor_batch_matmul(GenericTensorAccessorR const &lhs,
                                 GenericTensorAccessorR const &rhs,
                                 Allocator &output_allocator) {
  TensorShape output_shape =
      get_batch_matmul_output_shape(get_tensor_shape_for_accessor_r(lhs),
                                    get_tensor_shape_for_accessor_r(rhs));

  GenericTensorAccessorW output =
      output_allocator.allocate_tensor(output_shape);

  tensor_accessor_batch_matmul_to(lhs, rhs, output);

  return output;
}

void tensor_accessor_batch_matmul_to(GenericTensorAccessorR const &lhs,
                                     GenericTensorAccessorR const &rhs,
                                     GenericTensorAccessorW const &output) {
  TensorShape output_shape =
      get_batch_matmul_output_shape(get_tensor_shape_for_accessor_r(lhs),
                                    get_tensor_shape_for_accessor_r(rhs));

  positive_int batch_size =
      require_same(dim_at_idx(lhs.shape.dims, ff_dim_t{0_n}),
                   dim_at_idx(rhs.shape.dims, ff_dim_t{0_n}),
                   dim_at_idx(output.shape.dims, ff_dim_t{0_n}));

  positive_int lhs_rows = dim_at_idx(lhs.shape.dims, ff_dim_t{1_n});

  positive_int inner_dim_size =
      require_same(dim_at_idx(lhs.shape.dims, ff_dim_t{2_n}),
                   dim_at_idx(rhs.shape.dims, ff_dim_t{1_n}));

  positive_int rhs_columns = dim_at_idx(rhs.shape.dims, ff_dim_t{2_n});

  Allocator cpu_allocator = create_local_cpu_memory_allocator();
  GenericTensorAccessorR lhs_cpu =
      copy_tensor_accessor_r_to_cpu_if_necessary(lhs, cpu_allocator);
  GenericTensorAccessorR rhs_cpu =
      copy_tensor_accessor_r_to_cpu_if_necessary(rhs, cpu_allocator);
  GenericTensorAccessorW output_cpu =
      cpu_allocator.allocate_tensor(output_shape);

  for (nonnegative_int b : nonnegative_range(batch_size)) {
    for (nonnegative_int i : nonnegative_range(lhs_rows)) {
      for (nonnegative_int j : nonnegative_range(rhs_columns)) {
        float accum = 0.0f;
        for (nonnegative_int k : nonnegative_range(inner_dim_size)) {
          accum +=
              lhs_cpu.at<DataType::FLOAT>(TensorDimsCoord{FFOrdered{b, i, k}}) *
              rhs_cpu.at<DataType::FLOAT>(TensorDimsCoord{FFOrdered{b, k, j}});
        }
        output_cpu.at<DataType::FLOAT>(TensorDimsCoord{FFOrdered{b, i, j}}) =
            accum;
      }
    }
  }

  return copy_accessor_data_to_l_from_r(output, output_cpu);
}

static TensorShape get_concat_output_shape(TensorShape const &lhs,
                                           TensorShape const &rhs,
                                           ff_dim_t dim_idx) {
  std::map<ff_dim_t, positive_int> lhs_dim_sizes =
      map_from_ff_ordered(ff_ordered(lhs.dims));

  std::map<ff_dim_t, positive_int> rhs_dim_sizes =
      map_from_ff_ordered(ff_ordered(rhs.dims));

  std::set<ff_dim_t> output_dims =
      require_same(keys(lhs_dim_sizes), keys(rhs_dim_sizes));

  std::map<ff_dim_t, positive_int> output_dim_sizes =
      generate_map(output_dims, [&](ff_dim_t d) -> positive_int {
        positive_int lhs_dim_size = lhs_dim_sizes.at(d);
        positive_int rhs_dim_size = rhs_dim_sizes.at(d);

        if (d == dim_idx) {
          return lhs_dim_size + rhs_dim_size;
        } else {
          return require_same(lhs_dim_size, rhs_dim_size);
        }
      });

  return TensorShape{
      TensorDims{
          ff_ordered_from_map(output_dim_sizes),
      },
      /*data_type=*/require_same(lhs.data_type, rhs.data_type),
  };
}

template <DataType DT>
struct CPUConcatAccessors2 {
  void operator()(GenericTensorAccessorR const &lhs,
                  GenericTensorAccessorR const &rhs,
                  GenericTensorAccessorW &output,
                  ff_dim_t dim_idx) {

    using ValueType = real_type_t<DT>;

    positive_int lhs_concat_dim_size = dim_at_idx(lhs.shape.dims, dim_idx);

    ASSERT(lhs.device_type == DeviceType::CPU);
    ASSERT(rhs.device_type == DeviceType::CPU);
    ASSERT(output.device_type == DeviceType::CPU);

    auto get_output_value =
        [&](TensorDimsCoord const &output_coord) -> ValueType {
      nonnegative_int dim_idx_component =
          tensor_dims_coord_at_idx(output_coord, dim_idx);

      if (dim_idx_component < lhs_concat_dim_size) {
        return lhs.at<DT>(output_coord);
      } else {
        TensorDimsCoord rhs_coord = output_coord;
        tensor_dims_coord_at_idx(rhs_coord, dim_idx) = nonnegative_int{
            dim_idx_component.int_from_nonnegative_int() -
                lhs_concat_dim_size.int_from_positive_int(),
        };

        return rhs.at<DT>(rhs_coord);
      }
    };

    for (TensorDimsCoord const &coord :
         get_tensor_dims_coord_set(output.shape.dims)) {
      output.at<DT>(coord) = get_output_value(coord);
    }
  }
};

GenericTensorAccessorW
    tensor_accessor_binary_concat(GenericTensorAccessorR const &lhs,
                                  GenericTensorAccessorR const &rhs,
                                  ff_dim_t dim_idx,
                                  Allocator &output_allocator) {
  TensorShape output_shape =
      get_concat_output_shape(get_tensor_shape_for_accessor_r(lhs),
                              get_tensor_shape_for_accessor_r(rhs),
                              dim_idx);

  GenericTensorAccessorW output =
      output_allocator.allocate_tensor(output_shape);

  tensor_accessor_binary_concat_to(lhs, rhs, dim_idx, output);

  return output;
}

void tensor_accessor_binary_concat_to(GenericTensorAccessorR const &lhs,
                                      GenericTensorAccessorR const &rhs,
                                      ff_dim_t dim_idx,
                                      GenericTensorAccessorW const &output) {
  TensorShape output_shape =
      get_concat_output_shape(get_tensor_shape_for_accessor_r(lhs),
                              get_tensor_shape_for_accessor_r(rhs),
                              dim_idx);

  Allocator cpu_allocator = create_local_cpu_memory_allocator();
  GenericTensorAccessorR lhs_cpu =
      copy_tensor_accessor_r_to_cpu_if_necessary(lhs, cpu_allocator);
  GenericTensorAccessorR rhs_cpu =
      copy_tensor_accessor_r_to_cpu_if_necessary(rhs, cpu_allocator);
  GenericTensorAccessorW output_cpu =
      cpu_allocator.allocate_tensor(output_shape);

  DataType data_type = require_same(
      lhs.shape.data_type, rhs.shape.data_type, output_shape.data_type);

  DataTypeDispatch1<CPUConcatAccessors2>{}(
      data_type, lhs_cpu, rhs_cpu, output_cpu, dim_idx);

  return copy_accessor_data_to_l_from_r(output, output_cpu);
}

} // namespace FlexFlow
