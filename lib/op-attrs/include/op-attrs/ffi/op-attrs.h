#ifndef _FLEXFLOW_OPATTRS_INCLUDE_OPATTRS_FFI_OPATTRS_H
#define _FLEXFLOW_OPATTRS_INCLUDE_OPATTRS_FFI_OPATTRS_H

#include "utils/include/utils/ffi/utils.h"

FLEXFLOW_FFI_BEGIN()

typedef enum {
  FLEXFLOW_OPATTRS_STATUS_OK,
  FLEXFLOW_OPATTRS_ERROR_UNKNOWN,
} flexflow_opattrs_error_t;

typedef enum {
  FLEXFLOW_POOL_OP_MAX,
  FLEXFLOW_POOL_OP_AVG,
} flexflow_pool_op_t;

typedef enum {
  FLEXFLOW_AGGREGATE_OP_SUM,
  FLEXFLOW_AGGREGATE_OP_AVG,
} flexflow_aggregate_op_t;

FLEXFLOW_FFI_END()

#endif
