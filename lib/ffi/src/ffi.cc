#include "flexflow/flexflow.h"
#include "flexflow/op-attrs.h"
#include "flexflow/runtime.h"
#include "flexflow/utils.h"
#include "internal/pcg.h"

flexflow_error_t flexflow_status_is_ok(flexflow_error_t e, bool *result) {
  switch (e.error_source) {
    case FLEXFLOW_ERROR_SOURCE_RUNTIME:
    {
      flexflow_runtime_error_t err;
      CHECK_FLEXFLOW(flexflow_runtime_error_unwrap(e, &err));
      CHECK_FLEXFLOW(flexflow_runtime_error_is_ok(err, result));
    }
    case FLEXFLOW_ERROR_SOURCE_PCG:
    {
      flexflow_pcg_error_t err;
      CHECK_FLEXFLOW(flexflow_pcg_error_unwrap(e, &err));
      CHECK_FLEXFLOW(flexflow_pcg_error_is_ok(err, result));
    }
    case FLEXFLOW_ERROR_SOURCE_COMPILER:
    {
      flexflow_compiler_error_t err;
      CHECK_FLEXFLOW(flexflow_compiler_error_unwrap(e, &err));
      CHECK_FLEXFLOW(flexflow_compiler_error_is_ok(err, result));
    }
    case FLEXFLOW_ERROR_SOURCE_OPATTRS:
    {
      flexflow_opattrs_error_t err;
      CHECK_FLEXFLOW(flexflow_opattrs_error_unwrap(e, &err));
      CHECK_FLEXFLOW(flexflow_opattrs_error_is_ok(err, result));
    }
    case FLEXFLOW_ERROR_SOURCE_UTILS:
    {
      flexflow_utils_error_t err;
      CHECK_FLEXFLOW(flexflow_utils_error_unwrap(e, &err));
      CHECK_FLEXFLOW(flexflow_utils_error_is_ok(err, result));
    }
    default:
      return flexflow_utils_error_create(FLEXFLOW_UTILS_INVALID_ERROR_SOURCE);
  };

  return flexflow_utils_error_create(FLEXFLOW_UTILS_STATUS_OK);
}

flexflow_error_t flexflow_error_destroy(flexflow_error_t e) {
  switch (e.error_source) {
    case FLEXFLOW_ERROR_SOURCE_RUNTIME: {
      flexflow_runtime_error_t err;
      CHECK_FLEXFLOW(flexflow_runtime_error_unwrap(e, &err));
      return flexflow_runtime_error_destroy(err);
    }
    case FLEXFLOW_ERROR_SOURCE_PCG: {
      flexflow_pcg_error_t err;
      CHECK_FLEXFLOW(flexflow_pcg_error_unwrap(e, &err));
      return flexflow_pcg_error_destroy(err);
    }
    case FLEXFLOW_ERROR_SOURCE_COMPILER: {
      flexflow_compiler_error_t err;
      CHECK_FLEXFLOW(flexflow_compiler_error_unwrap(e, &err));
      return flexflow_compiler_error_destroy(err);
    }
    case FLEXFLOW_ERROR_SOURCE_OPATTRS: {
      flexflow_opattrs_error_t err;
      CHECK_FLEXFLOW(flexflow_opattrs_error_unwrap(e, &err));
      return flexflow_opattrs_error_destroy(err);
    }
    case FLEXFLOW_ERROR_SOURCE_UTILS: {
      return flexflow_utils_error_create(FLEXFLOW_UTILS_STATUS_OK);
    }
    default:
      return flexflow_utils_error_create(FLEXFLOW_UTILS_INVALID_ERROR_SOURCE);
  }
}
