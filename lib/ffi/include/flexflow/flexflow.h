#ifndef _FLEXFLOW_FFI_INCLUDE_FLEXFLOW_FLEXFLOW_H
#define _FLEXFLOW_FFI_INCLUDE_FLEXFLOW_FLEXFLOW_H

#include "flexflow/compiler.h"
#include "flexflow/op-attrs.h"
#include "flexflow/pcg.h"
#include "flexflow/runtime.h"
#include "flexflow/utils.h"
#include <stdio.h>
#include <stdlib.h>

FLEXFLOW_FFI_BEGIN();

#define CHECK_FLEXFLOW(status)                                                 \
  do {                                                                         \
    bool is_ok; \
    flexflow_status_is_ok(status, &is_ok);                                       \
    if (is_ok) { \
      char *error_msg; \
      assert(flexflow_status_is_ok(flexflow_get_error_string(status, &err_msg), &is_ok)); \
      fprintf(stderr,                                                          \
              "FlexFlow encountered an error at %s:%d : %s\n",                 \
              __FILE__,                                                        \
              __LINE__,                                                        \
              flexflow_get_error_string(status));                              \
      exit(flexflow_get_error_return_code(status));                            \
    }                                                                          \
  } while (0)

flexflow_error_t flexflow_status_is_ok(flexflow_error_t, bool *);
flexflow_error_t flexflow_get_error_string(flexflow_error_t, char *);

flexflow_error_t flexflow_error_destroy(flexflow_error_t);

FLEXFLOW_FFI_END();

#endif
