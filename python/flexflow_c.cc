#include "model.h"
#include "flexflow_c.h"

class FFCObjectWrapper {
public:
#define FF_NEW_OPAQUE_WRAPPER(T_, T)                                   \
  static T_ wrap(T t) {                                             \
    T_ t_;                                                          \
    t_.impl = static_cast<void *>(t);                               \
    return t_;                                                      \
  }                                                                 \
  static const T_ wrap_const(const T t) {                           \
    T_ t_;                                                          \
    t_.impl = const_cast<void *>(static_cast<const void *>(t));     \
    return t_;                                                      \
  }                                                                 \
  static T unwrap(T_ t_) {                                          \
    return static_cast<T>(t_.impl);                                 \
  }                                                                 \
  static const T unwrap_const(const T_ t_) {                        \
    return static_cast<const T>(t_.impl);                           \
  }
  
  FF_NEW_OPAQUE_WRAPPER(flexflow_config_t, FFConfig *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_model_t, FFModel *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_tensor_t, Tensor *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_sgd_optimizer_t, SGDOptimizer *);
};

// -----------------------------------------------------------------------
// FFConfig
// -----------------------------------------------------------------------

flexflow_config_t
flexflow_config_create()
{
  FFConfig *config = new FFConfig();
  Runtime *runtime = Runtime::get_runtime();
  config->lg_hlr = runtime;
  config->lg_ctx = Runtime::get_context();
  config->field_space = runtime->create_field_space(config->lg_ctx);
  return FFCObjectWrapper::wrap(config);
}

void
flexflow_config_destroy(
  flexflow_config_t handle_)
{
  FFConfig *handle = FFCObjectWrapper::unwrap(handle_);
  delete handle;
}

void
flexflow_config_parse_args(
  flexflow_config_t handle_,
  char** argv, 
  int argc)
{
  FFConfig *handle = FFCObjectWrapper::unwrap(handle_);
  handle->parse_args(argv, argc);  
}

int
flexflow_config_get_batch_size(
  flexflow_config_t handle_)
{
  FFConfig *handle = FFCObjectWrapper::unwrap(handle_);
  return handle->batchSize;
}

int
flexflow_config_get_workers_per_node(
  flexflow_config_t handle_)
{
  FFConfig *handle = FFCObjectWrapper::unwrap(handle_);
  return handle->workersPerNode;
}

int
flexflow_config_get_num_nodes(
  flexflow_config_t handle_)
{
  FFConfig *handle = FFCObjectWrapper::unwrap(handle_);
  return handle->numNodes;
}

int
flexflow_config_get_epochs(
  flexflow_config_t handle_)
{
  FFConfig *handle = FFCObjectWrapper::unwrap(handle_);
  return handle->epochs;
}

// -----------------------------------------------------------------------
// FFModel
// -----------------------------------------------------------------------

flexflow_model_t
flexflow_model_create(
  flexflow_config_t config_)
{
  FFConfig *config = FFCObjectWrapper::unwrap(config_);
  FFModel *model = new FFModel(*config);
  return FFCObjectWrapper::wrap(model);
}

void
flexflow_model_destroy(
  flexflow_model_t handle_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  delete handle;
}

void
flexflow_model_reset_metrics(
  flexflow_model_t handle_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  handle->reset_metrics();
}

void
flexflow_model_init_layers(
  flexflow_model_t handle_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  handle->init_layers();
}

void
flexflow_model_prefetch(
  flexflow_model_t handle_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  handle->prefetch();
}

void
flexflow_model_forward(
  flexflow_model_t handle_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  handle->forward();
}

void
flexflow_model_backward(
  flexflow_model_t handle_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  handle->backward();
}

void
flexflow_model_update(
  flexflow_model_t handle_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  handle->update();
}

void
flexflow_model_zero_gradients(
  flexflow_model_t handle_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  handle->zero_gradients();
}

// -----------------------------------------------------------------------
// Tensor
// -----------------------------------------------------------------------

flexflow_tensor_t
flexflow_tensor_4d_create(
  flexflow_model_t model_,
  const int* dims, 
  const char* pc_name, 
  DataType data_type, 
  bool create_grad /* true */)
{
  Tensor *tensor = new Tensor();
  FFModel *model = FFCObjectWrapper::unwrap(model_);
  Tensor tmp_tensor = model->create_tensor<4>(dims, pc_name, data_type, create_grad);
  *tensor = tmp_tensor;
  return FFCObjectWrapper::wrap(tensor);
}

void
flexflow_tensor_4d_destroy(
  flexflow_tensor_t handle_)
{
  Tensor *handle = FFCObjectWrapper::unwrap(handle_);
  delete handle;
}

// -----------------------------------------------------------------------
// SGDOptimizer
// -----------------------------------------------------------------------

flexflow_sgd_optimizer_t
flexflow_sgd_optimizer_create(
  flexflow_model_t model_,
  double lr, /* 0.01f */
  double momentum, /* 0.0f */
  bool nesterov, /* false */
  double weight_decay /* 0.0f */ )
{
  const FFModel *model = FFCObjectWrapper::unwrap_const(model_);
  SGDOptimizer *optimizer = new SGDOptimizer(model, lr, momentum, nesterov, weight_decay);
  return FFCObjectWrapper::wrap(optimizer);
}

void 
flexflow_sgd_optimizer_destroy(
  flexflow_sgd_optimizer_t handle_)
{
  SGDOptimizer *handle = FFCObjectWrapper::unwrap(handle_);
  delete handle;
}

