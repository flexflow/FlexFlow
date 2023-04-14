#ifndef _FLEXFLOW_KERNELS_ACCESSOR_H
#define _FLEXFLOW_KERNELS_ACCESSOR_H

#include "op-attrs/ffconst.h"
#include "kernels/ff_handler.h"
#include "array_shape.h"
#include <stdexcept>
#include "op-attrs/ffconst_utils.h"
#include "device.h"
#include "utils/exception.h"

namespace FlexFlow {

template <DataType> struct data_type_enum_to_class;

template <>
struct data_type_enum_to_class<DT_FLOAT> { using type = float; };

template <>
struct data_type_enum_to_class<DT_DOUBLE> { using type = double; };

template <>
struct data_type_enum_to_class<DT_INT32> { using type = int32_t; };

template <>
struct data_type_enum_to_class<DT_INT64> { using type = int64_t; };

template <>
struct data_type_enum_to_class<DT_HALF> { using type = half; };

template <>
struct data_type_enum_to_class<DT_BOOLEAN> { using type = bool; };

template <DataType DT, typename T>
typename data_type_enum_to_class<DT>::type cast_to(T t) {
  return (typename data_type_enum_to_class<DT>::type)t;
}

template <DataType DT> 
using real_type = typename data_type_enum_to_class<DT>::type;

size_t size_of(DataType);

class GenericTensorAccessorW : public use_visitable_eq<GenericTensorAccessorW> {
public:
  GenericTensorAccessorW() = delete;

  explicit GenericTensorAccessorW(DataType data_type, 
                                  ArrayShape const &shape,
                                  void *ptr);
  
  template <DataType DT>
  typename data_type_enum_to_class<DT>::type *get() const {
    if (this->data_type == DT) {
      return static_cast<real_type<DT> *>(this->ptr);
    } else {
      throw mk_runtime_error("Invalid access data type ({} != {})", this->data_type, DT);
    }
  }

  int32_t *get_int32_ptr() const;
  int64_t *get_int64_ptr() const;
  float *get_float_ptr() const;
  double *get_double_ptr() const;
  half *get_half_ptr() const;
  DataType data_type;
  ArrayShape shape;
  void *ptr;
};

class GenericTensorAccessorR : public use_visitable_eq<GenericTensorAccessorR> {
public:
  GenericTensorAccessorR() = delete;
  GenericTensorAccessorR(DataType data_type,
                         ArrayShape const &shape,
                         void const *ptr);
  explicit GenericTensorAccessorR(GenericTensorAccessorW const &);

  template <DataType DT>
  typename data_type_enum_to_class<DT>::type const *get() const {
    if (this->data_type == DT) {
      return static_cast<real_type<DT> const *>(this->ptr);
    } else {
      throw mk_runtime_error("Invalid access data type ({} != {})", this->data_type, DT);
    }
  }

  int32_t const *get_int32_ptr() const;
  int64_t const *get_int64_ptr() const;
  float const *get_float_ptr() const;
  double const *get_double_ptr() const;
  half const *get_half_ptr() const;
  DataType data_type;
  ArrayShape shape;
  void const *ptr;
};

}

VISITABLE_STRUCT(::FlexFlow::GenericTensorAccessorW, data_type, shape, ptr);
VISITABLE_STRUCT(::FlexFlow::GenericTensorAccessorR, data_type, shape, ptr);

#endif
