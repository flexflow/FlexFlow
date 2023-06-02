#ifndef _FLEXFLOW_FFI_INCLUDE_FLEXFLOW_FLEXFLOW_H
#define _FLEXFLOW_FFI_INCLUDE_FLEXFLOW_FLEXFLOW_H

#include "flexflow/runtime.h"

#define CHECK_FLEXFLOW(status) \
  do { \
    if (flexflow_status_is_ok(status)) { \
      printf("FlexFlow encountered an error: %s\n", flexflow_get_error_string(status)); \
    } \
  } while (0)

#endif
