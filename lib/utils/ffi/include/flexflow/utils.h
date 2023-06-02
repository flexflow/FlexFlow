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

#endif
