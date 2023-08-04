#ifndef _FLEXFLOW_UTILS_FFI_INTERNAL_INTERNAL_ERROR_HANDLING_H
#define _FLEXFLOW_UTILS_FFI_INTERNAL_INTERNAL_ERROR_HANDLING_H

#include "error.h"
#include "opaque.h"

template <typename Opaque>
flexflow_error_t handle_errors(Opaque *out, std::function<opaque_to_underlying_t<Opaque>()> const &f) {
  try {
    *out = new_opaque(f());
  } catch (flexflow_ffi_exception_t const &e) {
    return to_error(e);
  }

  return status_ok();
}


flexflow_error_t handle_errors(std::function<void()> const &f) {
  try {
    f();
  } catch (flexflow_ffi_exception_t const &e) {
    return to_error(e);
  }

  return status_ok();
}


#endif
