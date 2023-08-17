#ifndef _FLEXFLOW_UTILS_FFI_INTERNAL_INTERNAL_ERROR_H
#define _FLEXFLOW_UTILS_FFI_INTERNAL_INTERNAL_ERROR_H

#include "flexflow/utils.h"
#include "utils/type_traits_core.h"

#define RAISE_FLEXFLOW(status)                                                 \
  do {                                                                         \
    bool is_ok;                                                                \
    flexflow_status_is_ok(status, &is_ok);                                     \
    if (!is_ok) {                                                              \
      return status;                                                           \
    }                                                                          \
  } while (0)

struct flexflow_ffi_exception_t : public std::runtime_error {
  flexflow_ffi_exception_t(flexflow_error_t);

  flexflow_error_t err;
};

flexflow_ffi_exception_t make_utils_exception(flexflow_utils_error_code_t);

using flexflow_utils_exception_t = flexflow_ffi_exception_t;

flexflow_error_t to_error(flexflow_utils_exception_t const &);
flexflow_error_t status_ok();

template <typename T>
flexflow_error_t flexflow_error_wrap(flexflow_error_source_t error_source,
                                     T const &t) {
  static_assert(sizeof(T) < (FLEXFLOW_FFI_ERROR_BUF_SIZE * sizeof(char)), "");

  flexflow_error_t result;
  result.error_source = error_source;
  T *buf_ptr = static_cast<T *>(result.buf);
  *buf_ptr = t;

  return result;
}

template <typename T>
flexflow_error_t flexflow_error_unwrap(flexflow_error_t const &err,
                                       flexflow_error_source_t error_source,
                                       T *out) {
  static_assert(sizeof(T) < (FLEXFLOW_FFI_ERROR_BUF_SIZE * sizeof(char)), "");

  if (err.error_source != FLEXFLOW_ERROR_SOURCE_PCG) {
    return flexflow_utils_error_create(FLEXFLOW_UTILS_CAST_FAILED);
  }

  out->impl =
      reinterpret_cast<decltype(unwrap_opaque(std::declval<T>()))>(err.buf);
  return flexflow_utils_error_create(FLEXFLOW_UTILS_STATUS_OK);
}

#endif
