#include "kernels/accessor.h"

namespace FlexFlow {

int32_t *get_int32_ptr(GenericTensorAccessorW const &w) {
  return static_cast<int32_t *>(
      w.ptr.value()); // Note(lambda):we use static_cast, may have some problem
}

int64_t *get_int64_ptr(GenericTensorAccessorW const &w) {
  return static_cast<int64_t *>(w.ptr.value());
}
float *get_float_ptr(GenericTensorAccessorW const &w) {
  return static_cast<float *>(w.ptr.value());
}
double *get_double_ptr(GenericTensorAccessorW const &w) {
  return static_cast<double *>(w.ptr.value());
}
half *get_half_ptr(GenericTensorAccessorW const &w) {
  return static_cast<half *>(w.ptr.value());
}

} // namespace FlexFlow