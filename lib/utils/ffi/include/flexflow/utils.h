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
} flexflow_error_source_t;

typedef struct {
  flexflow_error_source_t error_source;
  int error_code;
} flexflow_error_t;

#endif
