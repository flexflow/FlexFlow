#ifndef _FLEXFLOW_FFI_INCLUDE_FLEXFLOW_FLEXFLOW_H
#define _FLEXFLOW_FFI_INCLUDE_FLEXFLOW_FLEXFLOW_H

#include "flexflow/runtime.h"
#include "flexflow/compiler.h"
#include "flexflow/pcg.h"
#include "flexflow/op-attrs.h"
#include <stdio.h>

#define CHECK_FLEXFLOW(status) \
  do { \
    if (flexflow_status_is_ok(status)) { \
      fprintf(stderr, "FlexFlow encountered an errorat %s:%d : %s\n", __FILE__, __LINE__, flexflow_get_error_string(status)); \
      exit(flexflow_get_error_return_code(status)); \
    } \
  } while (0)

bool flexflow_status_is_ok(flexflow_error_t);
char *flexflow_get_error_string(flexflow_error_t);
int flexflow_get_error_return_code(flexflow_error_t);

#endif
