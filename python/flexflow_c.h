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

// -----------------------------------------------------------------------
// FFConfig
// -----------------------------------------------------------------------

flexflow_config_t
flexflow_config_create();

void
flexflow_config_destroy(
  flexflow_config_t handle);

void
flexflow_config_parse_args(
  flexflow_config_t handle,
  char** argv, 
  int argc);
  
int
flexflow_config_get_batch_size(
  flexflow_config_t handle);

int
flexflow_config_get_workers_per_node(
  flexflow_config_t handle);

int
flexflow_config_get_num_nodes(
  flexflow_config_t handle);

int
flexflow_config_get_epochs(
  flexflow_config_t handle);

// -----------------------------------------------------------------------
// FFModel
// -----------------------------------------------------------------------

flexflow_model_t
flexflow_model_create(
  flexflow_config_t config);

void
flexflow_model_destroy(
  flexflow_model_t handle);

void
flexflow_model_reset_metrics(
  flexflow_model_t handle);

void
flexflow_model_init_layers(
  flexflow_model_t handle);

void
flexflow_model_prefetch(
  flexflow_model_t handle);

void
flexflow_model_forward(
  flexflow_model_t handle);

void
flexflow_model_backward(
  flexflow_model_t handle);

void
flexflow_model_update(
  flexflow_model_t handle);

void
flexflow_model_zero_gradients(
  flexflow_model_t handle);

flexflow_tensor_t
flexflow_model_add_conv2d(
  flexflow_model_t handle,
  char* name,
  const flexflow_tensor_t input,
  int out_channels,
  int kernel_h, int kernel_w,
  int stride_h, int stride_w,
  int padding_h, int padding_w,
  ActiMode activation /* AC_MODE_NONE */);
  
flexflow_tensor_t
flexflow_model_add_pool2d(
  flexflow_model_t handle,
  char* name,
  const flexflow_tensor_t input,
  int kernel_h, int kernel_w,
  int stride_h, int stride_w,
  int padding_h, int padding_w,
  PoolType type /* POOL_MAX */, 
  bool relu /* true */);
  
flexflow_tensor_t
flexflow_model_add_linear(
  flexflow_model_t handle,
  char* name,
  const flexflow_tensor_t input,
  int out_channels,
  ActiMode activation /* AC_MODE_NONE */,
  bool use_bias /* true */);
  
flexflow_tensor_t
flexflow_model_add_flat(
  flexflow_model_t handle,
  char* name,
  const flexflow_tensor_t input);
  
flexflow_tensor_t
flexflow_model_add_softmax(
  flexflow_model_t handle,
  char* name,
  const flexflow_tensor_t input);

// -----------------------------------------------------------------------
// Tensor
// -----------------------------------------------------------------------

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
  
// -----------------------------------------------------------------------
// SGDOptimizer
// -----------------------------------------------------------------------

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