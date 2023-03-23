#ifndef _FLEXFLOW_KERNELS_ACCESSOR_H
#define _FLEXFLOW_KERNELS_ACCESSOR_H

#include "op-attrs/ffconst.h"
#include "kernels/config.h"
#include "array_shape.h"
#include <stdexcept>
#include "op-attrs/ffconst_utils.h"

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

class GenericTensorAccessorW {
public:
  GenericTensorAccessorW();
  GenericTensorAccessorW(DataType data_type, 
                         ArrayShape const &shape,
                         void *ptr);
  
  template <DataType DT>
  typename data_type_enum_to_class<DT>::type *get() const;

  int32_t *get_int32_ptr() const;
  int64_t *get_int64_ptr() const;
  float *get_float_ptr() const;
  double *get_double_ptr() const;
  half *get_half_ptr() const;
  DataType data_type;
  ArrayShape shape;
  void *ptr;
};

class GenericTensorAccessorR {
public:
  GenericTensorAccessorR();
  GenericTensorAccessorR(DataType data_type,
                         ArrayShape const &shape,
                         void const *ptr);
  GenericTensorAccessorR(GenericTensorAccessorW const &acc);

  template <DataType DT>
  typename data_type_enum_to_class<DT>::type const *get() const;

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

#endif
