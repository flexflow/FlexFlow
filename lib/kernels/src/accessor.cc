#include "kernels/accessor.h"

namespace FlexFlow {

GenericTensorAccessorW::GenericTensorAccessorW(DataType data_type,
                                               ArrayShape const &shape,
                                               void *ptr)
    : data_type(data_type), shape(shape), ptr(ptr) 
{ }

int32_t *GenericTensorAccessorW::get_int32_ptr() const {
  return this->get<DT_INT32>();
}

int64_t *GenericTensorAccessorW::get_int64_ptr() const {
  return this->get<DT_INT64>();
}

float *GenericTensorAccessorW::get_float_ptr() const {
  return this->get<DT_FLOAT>();
}

double *GenericTensorAccessorW::get_double_ptr() const {
  return this->get<DT_DOUBLE>();
}

half *GenericTensorAccessorW::get_half_ptr() const {
  return this->get<DT_HALF>();
}

GenericTensorAccessorR::GenericTensorAccessorR(DataType data_type,
                                               ArrayShape const &shape,
                                               void const *ptr)
    : data_type(data_type), shape(shape), ptr(ptr)
{ }

GenericTensorAccessorR::GenericTensorAccessorR(GenericTensorAccessorW const &w) 
  : data_type(w.data_type), shape(w.shape), ptr(w.ptr)
{ }

int32_t const *GenericTensorAccessorR::get_int32_ptr() const {
  return this->get<DT_INT32>();
}

int64_t const *GenericTensorAccessorR::get_int64_ptr() const {
  return this->get<DT_INT64>();
}

float const *GenericTensorAccessorR::get_float_ptr() const {
  return this->get<DT_FLOAT>();
}

double const *GenericTensorAccessorR::get_double_ptr() const {
  return this->get<DT_DOUBLE>();
}

half const *GenericTensorAccessorR::get_half_ptr() const {
  return this->get<DT_HALF>();
}


}
