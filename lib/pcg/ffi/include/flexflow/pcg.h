#ifndef _FLEXFLOW_PCG_FFI_INCLUDE_FLEXFLOW_PCG_H
#define _FLEXFLOW_PCG_FFI_INCLUDE_FLEXFLOW_PCG_H

#include "flexflow/utils.h"

FLEXFLOW_FFI_BEGIN()

typedef enum {
  FLEXFLOW_PCG_STATUS_OK,
  FLEXFLOW_PCG_ERROR_UNKNOWN,
} flexflow_pcg_error_t;

FF_NEW_OPAQUE_TYPE(flexflow_computation_graph_t);
FF_NEW_OPAQUE_TYPE(flexflow_parallel_computation_graph_t);

FLEXFLOW_FFI_END()

#endif
