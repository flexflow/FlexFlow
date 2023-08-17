#include "internal/error_handling.h" 

flexflow_error_t handle_errors(std::function<void()> const &f) {
  try {
    f();
  } catch (flexflow_ffi_exception_t const &e) {
    return to_error(e);
  }

  return status_ok();
}


