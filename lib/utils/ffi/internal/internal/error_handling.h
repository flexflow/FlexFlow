#ifndef _FLEXFLOW_UTILS_FFI_INTERNAL_INTERNAL_ERROR_HANDLING_H
#define _FLEXFLOW_UTILS_FFI_INTERNAL_INTERNAL_ERROR_HANDLING_H

#include "error.h"
#include "opaque.h"

flexflow_error_t handle_errors(std::function<void()> const &f);

template <typename Opaque>
flexflow_error_t
    handle_errors(Opaque *out,
                  std::function<opaque_to_underlying_t<Opaque>()> const &f) {
  return handle_errors([&] { *out = new_opaque(f()); });
}

#endif
