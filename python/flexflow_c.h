#ifndef __FLEXFLOW_C_H__
#define __FLEXFLOW_C_H__

#include "ffconst.h"

#ifdef __cplusplus
extern "C" {
#endif
  
#define FF_NEW_OPAQUE_TYPE(T) typedef struct T { void *impl; } T

FF_NEW_OPAQUE_TYPE(flexflow_config_t);
FF_NEW_OPAQUE_TYPE(flexflow_model_t);
FF_NEW_OPAQUE_TYPE(flexflow_tensor_t);
FF_NEW_OPAQUE_TYPE(flexflow_sgd_optimizer_t);

flexflow_model_t
flexflow_model_create(
  flexflow_config_t config);

void
flexflow_model_destroy(
  flexflow_model_t handle);

flexflow_tensor_t
flexflow_tensor_4d_create(
  flexflow_model_t model,
  const int* dims, 
  const char* pc_name, 
  DataType data_type, 
  bool create_grad /* true */);

void
flexflow_tensor_4d_destroy(
  flexflow_tensor_t handle);
  
flexflow_sgd_optimizer_t
flexflow_sgd_optimizer_create(
  flexflow_model_t model,
  double lr, /* 0.01f */
  double momentum, /* 0.0f */
  bool nesterov, /* false */
  double weight_decay /* 0.0f */ );

void 
flexflow_sgd_optimizer_destroy(
  flexflow_sgd_optimizer_t handle);


#ifdef __cplusplus
}
#endif

#endif // __FLEXFLOW_C_H__