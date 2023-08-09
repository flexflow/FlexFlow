#ifndef _FLEXFLOW_KERNELS_ACCESSOR_H
#define _FLEXFLOW_KERNELS_ACCESSOR_H

#include "array_shape.h"
#include "device.h"
#include "kernels/ff_handle.h"
#include "op-attrs/datatype.h"
#include "utils/exception.h"
#include "utils/variant.h"

namespace FlexFlow {

template <DataType>
struct data_type_enum_to_class;

template <>
struct data_type_enum_to_class<DataType::FLOAT> {
  using type = float;
};

template <>
struct data_type_enum_to_class<DataType::DOUBLE> {
  using type = double;
};

template <>
struct data_type_enum_to_class<DataType::INT32> {
  using type = int32_t;
};

template <>
struct data_type_enum_to_class<DataType::INT64> {
  using type = int64_t;
};

template <>
struct data_type_enum_to_class<DataType::HALF> {
  using type = half;
};

template <>
struct data_type_enum_to_class<DataType::BOOL> {
  using type = bool;
};

template <DataType DT, typename T>
typename data_type_enum_to_class<DT>::type cast_to(T t) {
  return (typename data_type_enum_to_class<DT>::type)t;
}

template <DataType DT>
using real_type = typename data_type_enum_to_class<DT>::type;

using DataTypeValue = variant<real_type<DataType::FLOAT>,
                              real_type<DataType::DOUBLE>,
                              real_type<DataType::INT32>,
                              real_type<DataType::INT64>,
                              real_type<DataType::HALF>,
                              real_type<DataType::BOOL>>;

size_t size_of(DataType);

class GenericTensorAccessorW
    : public use_visitable_cmp<GenericTensorAccessorW> {
public:
  GenericTensorAccessorW() = delete;

  explicit GenericTensorAccessorW(DataType data_type,
                                  ArrayShape const &shape,
                                  void *ptr);

  /* template <DataType DT> */
  /* typename data_type_enum_to_class<DT>::type *get() const { */
  /*   if (this->data_type == DT) { */
  /*     return static_cast<real_type<DT> *>(this->ptr); */
  /*   } else { */
  /*     throw mk_runtime_error("Invalid access data type ({} != {})",
   * this->data_type, DT); */
  /*   } */
  /* } */

  /* int32_t *get_int32_ptr() const; */
  /* int64_t *get_int64_ptr() const; */
  /* float *get_float_ptr() const; */
  /* double *get_double_ptr() const; */
  /* half *get_half_ptr() const; */
  DataType data_type;
  ArrayShape shape;
  void *ptr;
};

class GenericTensorAccessorR
    : public use_visitable_cmp<GenericTensorAccessorR> {
public:
  GenericTensorAccessorR() = delete;
  GenericTensorAccessorR(DataType data_type,
                         ArrayShape const &shape,
                         void const *ptr);
  explicit GenericTensorAccessorR(GenericTensorAccessorW const &);

  /* template <DataType DT> */
  /* typename data_type_enum_to_class<DT>::type const *get() const { */
  /*   if (this->data_type == DT) { */
  /*     return static_cast<real_type<DT> const *>(this->ptr); */
  /*   } else { */
  /*     throw mk_runtime_error("Invalid access data type ({} != {})",
   * this->data_type, DT); */
  /*   } */
  /* } */

  /* int32_t const *get_int32_ptr() const; */
  /* int64_t const *get_int64_ptr() const; */
  /* float const *get_float_ptr() const; */
  /* double const *get_double_ptr() const; */
  /* half const *get_half_ptr() const; */
  DataType data_type;
  ArrayShape shape;
  void const *ptr;
};

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

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::GenericTensorAccessorW, data_type, shape, ptr);
VISITABLE_STRUCT(::FlexFlow::GenericTensorAccessorR, data_type, shape, ptr);

namespace FlexFlow {
static_assert(is_well_behaved_value_type<GenericTensorAccessorR>::value, "");
static_assert(is_well_behaved_value_type<GenericTensorAccessorW>::value, "");
} // namespace FlexFlow

#endif
