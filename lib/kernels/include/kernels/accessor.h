#ifndef _FLEXFLOW_KERNELS_ACCESSOR_H
#define _FLEXFLOW_KERNELS_ACCESSOR_H

#include "array_shape.h"
#include "device.h"
#include "kernels/ff_handle.h"
#include "op-attrs/datatype.h"
#include "utils/exception.h"
#include "utils/required.h"
#include "utils/variant.h"

namespace FlexFlow {

class GenericTensorAccessorW {
public:
  template <DataType DT>
  typename data_type_enum_to_class<DT>::type *get() const {
    if (this->data_type == DT) {
      return static_cast<real_type<DT> *>(this->ptr);
    } else {
      throw mk_runtime_error(
          "Invalid access data type ({} != {})", this->data_type, DT);
    }
  }

  int32_t *get_int32_ptr() const;
  int64_t *get_int64_ptr() const;
  float *get_float_ptr() const;
  double *get_double_ptr() const;
  half *get_half_ptr() const;

public:
  DataType data_type;
  ArrayShape shape;
  req<void *> ptr;
};
FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(GenericTensorAccessorW,
                                             data_type,
                                             shape,
                                             ptr);

class GenericTensorAccessorR {
public:
  template <DataType DT>
  typename data_type_enum_to_class<DT>::type const *get() const {
    if (this->data_type == DT) {
      return static_cast<real_type<DT> const *>(this->ptr);
    } else {
      throw mk_runtime_error(
          "Invalid access data type ({} != {})", this->data_type, DT);
    }
  }

  int32_t const *get_int32_ptr() const;
  int64_t const *get_int64_ptr() const;
  float const *get_float_ptr() const;
  double const *get_double_ptr() const;
  half const *get_half_ptr() const;

public:
  DataType data_type;
  ArrayShape shape;
  req<void const *> ptr;
};
FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(GenericTensorAccessorR,
                                             data_type,
                                             shape,
                                             ptr);

int32_t *get_int32_ptr(GenericTensorAccessorW const &);
int64_t *get_int64_ptr(GenericTensorAccessorW const &);
float *get_float_ptr(GenericTensorAccessorW const &);
double *get_double_ptr(GenericTensorAccessorW const &);
half *get_half_ptr(GenericTensorAccessorW const &);
std::vector<int32_t *>
    get_int32_ptrs(std::vector<GenericTensorAccessorW> const &);
std::vector<int64_t *>
    get_int64_ptrs(std::vector<GenericTensorAccessorW> const &);
std::vector<float *>
    get_float_ptrs(std::vector<GenericTensorAccessorW> const &);
std::vector<double *>
    get_double_ptrs(std::vector<GenericTensorAccessorW> const &);
std::vector<half *> get_half_ptrs(std::vector<GenericTensorAccessorW> const &);

static_assert(is_fmtable<req<DataType> const &>::value, "");

template <DataType DT>
typename data_type_enum_to_class<DT>::type *
    get(GenericTensorAccessorW const &a) {
  if (a.data_type == DT) {
    return static_cast<real_type<DT> *>(a.ptr);
  } else {
    throw mk_runtime_error(
        "Invalid access data type ({} != {})", a.data_type, DT);
  }
}

template <DataType DT>
std::vector<real_type<DT> *>
    get(std::vector<GenericTensorAccessorW> const &accs) {
  std::vector<real_type<DT> *> out;
  for (auto acc : accs) {
    out.push_back(get<DT>(acc));
  }
  return out;
}

template <DataType DT>
typename data_type_enum_to_class<DT>::type const *
    get(GenericTensorAccessorR const &a) {
  if (a.data_type == DT) {
    return static_cast<real_type<DT> const *>(a.ptr);
  } else {
    throw mk_runtime_error(
        "Invalid access data type ({} != {})", a.data_type, DT);
  }
}

int32_t const *get_int32_ptr(GenericTensorAccessorR const &);
int64_t const *get_int64_ptr(GenericTensorAccessorR const &);
float const *get_float_ptr(GenericTensorAccessorR const &);
double const *get_double_ptr(GenericTensorAccessorR const &);
half const *get_half_ptr(GenericTensorAccessorR const &);
std::vector<int32_t const *>
    get_int32_ptrs(std::vector<GenericTensorAccessorR> const &);
std::vector<int64_t const *>
    get_int64_ptrs(std::vector<GenericTensorAccessorR> const &);
std::vector<float const *>
    get_float_ptrs(std::vector<GenericTensorAccessorR> const &);
std::vector<double const *>
    get_double_ptrs(std::vector<GenericTensorAccessorR> const &);
std::vector<half const *>
    get_half_ptrs(std::vector<GenericTensorAccessorR> const &);

template <DataType DT>
std::vector<real_type<DT> const *>
    get(std::vector<GenericTensorAccessorR> const &accs) {
  std::vector<real_type<DT> const *> out;
  for (auto acc : accs) {
    out.push_back(get<DT>(acc));
  }
  return out;
}

GenericTensorAccessorR read_only_accessor_from_write_accessor(
    GenericTensorAccessorW const &write_accessor);

} // namespace FlexFlow

namespace FlexFlow {
static_assert(is_well_behaved_value_type_no_hash<GenericTensorAccessorR>::value,
              "");
static_assert(is_well_behaved_value_type_no_hash<GenericTensorAccessorW>::value,
              "");
} // namespace FlexFlow

#endif
