#include "internal/error.h"

flexflow_error_t status_ok() {
  return flexflow_utils_error_create(FLEXFLOW_UTILS_STATUS_OK);
}
