#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_FFI_UTILS_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_FFI_UTILS_H

#ifdef __cplusplus
#define FLEXFLOW_FFI_BEGIN() extern "C" {
#else
#define FLEXFLOW_FFI_BEGIN()
#endif

#ifdef __cplusplus
#define FLEXFLOW_FFI_END() }
#else
#define FLEXFLOW_FFI_END()
#endif

#define FF_NEW_OPAQUE_TYPE(T)                                                  \
  typedef struct T {                                                           \
    void *impl;                                                                \
  } T

typedef enum {
  FLEXFLOW_ERROR_SOURCE_RUNTIME,
  FLEXFLOW_ERROR_SOURCE_PCG,
  FLEXFLOW_ERROR_SOURCE_COMPILER,
  FLEXFLOW_ERROR_SOURCE_OPATTRS,
  FLEXFLOW_ERROR_SOURCE_UTILS,
} flexflow_error_source_t;

typedef enum {
  FLEXFLOW_UTILS_STATUS_OK, 
  FLEXFLOW_UTILS_DEALLOCATION_FAILED,
  FLEXFLOW_UTILS_ALLOCATION_FAILED,
  FLEXFLOW_UTILS_CAST_FAILED,
  FLEXFLOW_UTILS_INVALID_ERROR_SOURCE,
  FLEXFLOW_UTILS_UNEXPECTED_NULLPTR_IN_OPAQUE_HANDLE,
} flexflow_utils_error_code_t;

typedef struct {
  flexflow_utils_error_code_t err_code;
} flexflow_utils_error_t;

#define FLEXFLOW_FFI_ERROR_BUF_SIZE 24

typedef struct {
  flexflow_error_source_t error_source;
  char buf[FLEXFLOW_FFI_ERROR_BUF_SIZE];
} flexflow_error_t;

flexflow_error_t flexflow_utils_error_is_ok(flexflow_utils_error_t, bool *);
flexflow_error_t flexflow_utils_error_create(flexflow_utils_error_code_t);
flexflow_error_t flexflow_utils_error_unwrap(flexflow_error_t, flexflow_utils_error_t *);

flexflow_error_t flexflow_status_is_ok(flexflow_error_t, bool *);

#endif
