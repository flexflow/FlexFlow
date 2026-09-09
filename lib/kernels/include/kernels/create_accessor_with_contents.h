#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_CREATE_ACCESSOR_WITH_CONTENTS_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_CREATE_ACCESSOR_WITH_CONTENTS_H

#include "kernels/accessor.h"
#include "kernels/allocation.h"
#include "kernels/local_cpu_allocator.h"
#include "utils/containers/require_all_same1.h"
#include "utils/nonnegative_int/nonnegative_range.h"

namespace FlexFlow {

template <typename T>
GenericTensorAccessorW
    create_1d_accessor_w_with_contents(std::vector<T> const &contents,
                                       Allocator &allocator) {
  positive_int ncols = positive_int{num_elements(contents)};

  TensorShape shape = TensorShape{
      TensorDims{FFOrdered{ncols}},
      type_to_data_type_enum_v<T>,
  };

  Allocator cpu_allocator = create_local_cpu_memory_allocator();
  GenericTensorAccessorW cpu_accessor = cpu_allocator.allocate_tensor(shape);

  for (nonnegative_int col_idx :
       nonnegative_range(ncols.nonnegative_int_from_positive_int())) {
    cpu_accessor.at<type_to_data_type_enum_v<T>>(TensorDimsCoord{
        FFOrdered{col_idx}}) = contents.at(col_idx.unwrap_nonnegative());
  }

  GenericTensorAccessorW result = allocator.allocate_tensor(shape);
  copy_accessor_data_to_l_from_r(
      result, read_only_accessor_from_write_accessor(cpu_accessor));

  return result;
}

template <typename T>
GenericTensorAccessorW create_2d_accessor_w_with_contents(
    std::vector<std::vector<T>> const &contents, Allocator &allocator) {
  positive_int nrows = positive_int{num_elements(contents)};

  positive_int ncols =
      require_all_same1(transform(contents, [](std::vector<T> const &row) {
        return positive_int{num_elements(row)};
      }));

  TensorShape shape = TensorShape{
      TensorDims{FFOrdered{nrows, ncols}},
      type_to_data_type_enum_v<T>,
  };

  Allocator cpu_allocator = create_local_cpu_memory_allocator();
  GenericTensorAccessorW cpu_accessor = cpu_allocator.allocate_tensor(shape);

  for (nonnegative_int row_idx :
       nonnegative_range(nrows.nonnegative_int_from_positive_int())) {
    for (nonnegative_int col_idx :
         nonnegative_range(ncols.nonnegative_int_from_positive_int())) {
      cpu_accessor.at<type_to_data_type_enum_v<T>>(
          TensorDimsCoord{FFOrdered{row_idx, col_idx}}) =
          contents.at(row_idx.unwrap_nonnegative())
              .at(col_idx.unwrap_nonnegative());
    }
  }

  GenericTensorAccessorW result = allocator.allocate_tensor(shape);
  copy_accessor_data_to_l_from_r(
      result, read_only_accessor_from_write_accessor(cpu_accessor));

  return result;
}

template <typename T>
GenericTensorAccessorW create_3d_accessor_w_with_contents(
    std::vector<std::vector<std::vector<T>>> const &contents,
    Allocator &allocator) {
  positive_int dim0_size = positive_int{num_elements(contents)};

  positive_int dim1_size = require_all_same1(
      transform(contents, [](std::vector<std::vector<T>> const &m) {
        return positive_int{num_elements(m)};
      }));

  positive_int dim2_size = require_all_same1(
      transform(contents, [](std::vector<std::vector<T>> const &m) {
        return require_all_same1(transform(m, [](std::vector<T> const &vec) {
          return positive_int{num_elements(vec)};
        }));
      }));

  TensorShape shape = TensorShape{
      TensorDims{FFOrdered{dim0_size, dim1_size, dim2_size}},
      type_to_data_type_enum_v<T>,
  };

  Allocator cpu_allocator = create_local_cpu_memory_allocator();
  GenericTensorAccessorW cpu_accessor = cpu_allocator.allocate_tensor(shape);

  for (nonnegative_int dim0_idx :
       nonnegative_range(dim0_size.nonnegative_int_from_positive_int())) {
    for (nonnegative_int dim1_idx :
         nonnegative_range(dim1_size.nonnegative_int_from_positive_int())) {
      for (nonnegative_int dim2_idx :
           nonnegative_range(dim2_size.nonnegative_int_from_positive_int())) {
        cpu_accessor.at<type_to_data_type_enum_v<T>>(
            TensorDimsCoord{FFOrdered{dim0_idx, dim1_idx, dim2_idx}}) =
            contents.at(dim0_idx.unwrap_nonnegative())
                .at(dim1_idx.unwrap_nonnegative())
                .at(dim2_idx.unwrap_nonnegative());
      }
    }
  }

  GenericTensorAccessorW result = allocator.allocate_tensor(shape);
  copy_accessor_data_to_l_from_r(
      result, read_only_accessor_from_write_accessor(cpu_accessor));

  return result;
}

template <typename T>
GenericTensorAccessorW create_4d_accessor_w_with_contents(
    std::vector<std::vector<std::vector<std::vector<T>>>> const &contents,
    Allocator &allocator) {
  positive_int dim0_size = positive_int{num_elements(contents)};

  positive_int dim1_size = require_all_same1(transform(
      contents, [](std::vector<std::vector<std::vector<T>>> const &t) {
        return positive_int{num_elements(t)};
      }));

  positive_int dim2_size = require_all_same1(transform(
      contents, [](std::vector<std::vector<std::vector<T>>> const &m) {
        return require_all_same1(
            transform(m, [](std::vector<std::vector<T>> const &vec) {
              return positive_int{num_elements(vec)};
            }));
      }));

  positive_int dim3_size = require_all_same1(transform(
      contents, [](std::vector<std::vector<std::vector<T>>> const &t) {
        return require_all_same1(
            transform(t, [](std::vector<std::vector<T>> const &mat) {
              return require_all_same1(
                  transform(mat, [](std::vector<T> const &vec) {
                    return positive_int{num_elements(vec)};
                  }));
            }));
      }));

  TensorShape shape = TensorShape{
      TensorDims{FFOrdered{dim0_size, dim1_size, dim2_size, dim3_size}},
      type_to_data_type_enum_v<T>,
  };

  Allocator cpu_allocator = create_local_cpu_memory_allocator();
  GenericTensorAccessorW cpu_accessor = cpu_allocator.allocate_tensor(shape);

  for (nonnegative_int dim0_idx :
       nonnegative_range(dim0_size.nonnegative_int_from_positive_int())) {
    for (nonnegative_int dim1_idx :
         nonnegative_range(dim1_size.nonnegative_int_from_positive_int())) {
      for (nonnegative_int dim2_idx :
           nonnegative_range(dim2_size.nonnegative_int_from_positive_int())) {
        for (nonnegative_int dim3_idx :
             nonnegative_range(dim3_size.nonnegative_int_from_positive_int())) {
          cpu_accessor.at<type_to_data_type_enum_v<T>>(TensorDimsCoord{
              FFOrdered{dim0_idx, dim1_idx, dim2_idx, dim3_idx}}) =
              contents.at(dim0_idx.unwrap_nonnegative())
                  .at(dim1_idx.unwrap_nonnegative())
                  .at(dim2_idx.unwrap_nonnegative())
                  .at(dim3_idx.unwrap_nonnegative());
        }
      }
    }
  }

  GenericTensorAccessorW result = allocator.allocate_tensor(shape);
  copy_accessor_data_to_l_from_r(
      result, read_only_accessor_from_write_accessor(cpu_accessor));

  return result;
}

template <typename T>
GenericTensorAccessorR
    create_1d_accessor_r_with_contents(std::vector<T> const &contents,
                                       Allocator &allocator) {
  return read_only_accessor_from_write_accessor(
      create_1d_accessor_w_with_contents(contents, allocator));
}

template <typename T>
GenericTensorAccessorR create_2d_accessor_r_with_contents(
    std::vector<std::vector<T>> const &contents, Allocator &allocator) {
  return read_only_accessor_from_write_accessor(
      create_2d_accessor_w_with_contents(contents, allocator));
}

template <typename T>
GenericTensorAccessorR create_3d_accessor_r_with_contents(
    std::vector<std::vector<std::vector<T>>> const &contents,
    Allocator &allocator) {
  return read_only_accessor_from_write_accessor(
      create_3d_accessor_w_with_contents(contents, allocator));
}

template <typename T>
GenericTensorAccessorR create_4d_accessor_r_with_contents(
    std::vector<std::vector<std::vector<std::vector<T>>>> const &contents,
    Allocator &allocator) {
  return read_only_accessor_from_write_accessor(
      create_4d_accessor_w_with_contents(contents, allocator));
}

} // namespace FlexFlow

#endif
