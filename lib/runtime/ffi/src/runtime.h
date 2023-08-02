#ifndef _FLEXFLOW_RUNTIME_FFI_SRC_RUNTIME_H
#define _FLEXFLOW_RUNTIME_FFI_SRC_RUNTIME_H

#include "flexflow/runtime.h"
#include "utils/exception.h"

struct internal_flexflow_runtime_error_t {
  flexflow_runtime_error_code_t err_code;
};

flexflow_error_t flexflow_runtime_error_create(internal_flexflow_runtime_error_t const &);


#endif
