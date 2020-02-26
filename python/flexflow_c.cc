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
  printf("new FFConfig %p\n", config);
  return FFCObjectWrapper::wrap(config);
}

void
flexflow_config_destroy(
  flexflow_config_t handle_)
{
  FFConfig *handle = FFCObjectWrapper::unwrap(handle_);
  printf("delete FFConfig %p\n", handle);
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

void
flexflow_config_parse_default_args(
  flexflow_config_t handle_)
{
  FFConfig *handle = FFCObjectWrapper::unwrap(handle_);
  const InputArgs &command_args = Runtime::get_input_args();
  char **argv = command_args.argv;
  int argc = command_args.argc;
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
  printf("new FFModel %p\n", model);
  return FFCObjectWrapper::wrap(model);
}

void
flexflow_model_destroy(
  flexflow_model_t handle_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  printf("delete FFModel %p\n", handle); 
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

flexflow_tensor_t
flexflow_model_add_conv2d(
  flexflow_model_t handle_,
  const char* name,
  const flexflow_tensor_t input_,
  int out_channels,
  int kernel_h, int kernel_w,
  int stride_h, int stride_w,
  int padding_h, int padding_w,
  enum ActiMode activation /* AC_MODE_NONE */)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor *input = FFCObjectWrapper::unwrap_const(input_);
  Tensor *tensor = new Tensor();
  *tensor = handle->conv2d(name, *input, out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, activation);
  printf("conv2d new Tensor 4D %p\n", tensor);
  return FFCObjectWrapper::wrap(tensor);   
}

flexflow_tensor_t
flexflow_model_add_pool2d(
  flexflow_model_t handle_,
  const char* name,
  const flexflow_tensor_t input_,
  int kernel_h, int kernel_w,
  int stride_h, int stride_w,
  int padding_h, int padding_w,
  enum PoolType type /* POOL_MAX */, 
  bool relu /* true */)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor *input = FFCObjectWrapper::unwrap_const(input_);
  Tensor *tensor = new Tensor();
  *tensor = handle->pool2d(name, *input, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, type, relu);
  printf("pool2d new Tensor 4D %p\n", tensor);
  return FFCObjectWrapper::wrap(tensor); 
}

flexflow_tensor_t
flexflow_model_add_linear(
  flexflow_model_t handle_,
  const char* name,
  const flexflow_tensor_t input_,
  int out_channels,
  enum ActiMode activation /* AC_MODE_NONE */,
  bool use_bias /* true */)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor *input = FFCObjectWrapper::unwrap_const(input_);
  Tensor *tensor = new Tensor();
  *tensor = handle->linear(name, *input, out_channels, activation, use_bias);
  printf("linear new Tensor 4D %p\n", tensor);
  return FFCObjectWrapper::wrap(tensor); 
}

flexflow_tensor_t
flexflow_model_add_concat(
  flexflow_model_t handle_,
  const char* name,
  int n,
  flexflow_tensor_t* input_,
  int axis)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor *tensor = new Tensor();
  std::vector<Tensor> input_vec;
  for (int i = 0; i < n; i++ ) {
    Tensor *t = FFCObjectWrapper::unwrap(input_[i]);
    input_vec.push_back(*t);
  }
  *tensor = handle->concat(name, n, input_vec.data(), axis);
  printf("concat new Tensor 4D %p\n", tensor);
  return FFCObjectWrapper::wrap(tensor); 
}

flexflow_tensor_t
flexflow_model_add_flat(
  flexflow_model_t handle_,
  const char* name,
  const flexflow_tensor_t input_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor *input = FFCObjectWrapper::unwrap_const(input_);
  Tensor *tensor = new Tensor();
  *tensor = handle->flat(name, *input);
  printf("flat new Tensor 4D %p\n", tensor);
  return FFCObjectWrapper::wrap(tensor);  
}

flexflow_tensor_t
flexflow_model_add_softmax(
  flexflow_model_t handle_,
  char* name,
  const flexflow_tensor_t input_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor *input = FFCObjectWrapper::unwrap_const(input_);
  Tensor *tensor = new Tensor();
  *tensor = handle->softmax(name, *input);
  printf("softmax new Tensor 4D %p\n", tensor);
  return FFCObjectWrapper::wrap(tensor);   
}

void
flexflow_model_set_sgd_optimizer(
  flexflow_model_t handle_, 
  flexflow_sgd_optimizer_t optimizer_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  SGDOptimizer *optimizer = FFCObjectWrapper::unwrap(optimizer_);
  handle->optimizer = static_cast<Optimizer *>(optimizer);
}

// -----------------------------------------------------------------------
// Tensor
// -----------------------------------------------------------------------

flexflow_tensor_t
flexflow_tensor_4d_create(
  flexflow_model_t model_,
  const int* dims, 
  const char* pc_name, 
  enum DataType data_type, 
  bool create_grad /* true */)
{
  Tensor *tensor = new Tensor();
  FFModel *model = FFCObjectWrapper::unwrap(model_);
  *tensor = model->create_tensor<4>(dims, pc_name, data_type, create_grad);
  printf("new Tensor 4D %p\n", tensor);
  return FFCObjectWrapper::wrap(tensor);
}

void
flexflow_tensor_4d_destroy(
  flexflow_tensor_t handle_)
{
  Tensor *handle = FFCObjectWrapper::unwrap(handle_);
  printf("delete Tensor 4D %p\n", handle);
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

