#ifndef _FLEXFLOW_KERNELS_ACCESSOR_H
#define _FLEXFLOW_KERNELS_ACCESSOR_H

#include "domain.h"
#include "op-meta/ffconst.h"
#include "kernels/config.h"

namespace FlexFlow {

class GenericTensorAccessorW {
public:
  GenericTensorAccessorW();
  GenericTensorAccessorW(DataType data_type, Legion::Domain domain, void *ptr);
  int32_t *get_int32_ptr() const;
  int64_t *get_int64_ptr() const;
  float *get_float_ptr() const;
  double *get_double_ptr() const;
  half *get_half_ptr() const;
  DataType data_type;
  NaryShape domain;
  void *ptr;
};

class GenericTensorAccessorR {
public:
  GenericTensorAccessorR();
  GenericTensorAccessorR(DataType data_type,
                         Legion::Domain domain,
                         void const *ptr);
  GenericTensorAccessorR(GenericTensorAccessorW const &acc);

  int32_t const *get_int32_ptr() const;
  int64_t const *get_int64_ptr() const;
  float const *get_float_ptr() const;
  double const *get_double_ptr() const;
  half const *get_half_ptr() const;
  DataType data_type;
  Legion::Domain domain;
  void const *ptr;
};

}

#endif
