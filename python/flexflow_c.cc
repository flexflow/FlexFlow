/* Copyright 2020 Stanford, Los Alamos National Laboratory
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "flexflow_dataloader.h"
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
  FF_NEW_OPAQUE_WRAPPER(flexflow_adam_optimizer_t, AdamOptimizer *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_initializer_t, Initializer *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_glorot_uniform_initializer_t, GlorotUniform *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_zero_initializer_t, ZeroInitializer *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_uniform_initializer_t, UniformInitializer *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_norm_initializer_t, NormInitializer *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_op_t, Op *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_parameter_t, Parameter *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_perf_metrics_t, PerfMetrics *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_net_config_t, NetConfig *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_dataloader_4d_t, ImgDataLoader4D *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_dataloader_2d_t, ImgDataLoader2D *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_single_dataloader_t, SingleDataLoader *);
};

Logger ffc_log("flexflow_c");

#ifdef FF_DEBUG
#define DEBUG_PRINT(...) ffc_log.print(__VA_ARGS__)
#else
#define DEBUG_PRINT(...)
#endif

// -----------------------------------------------------------------------
// FFConfig
// -----------------------------------------------------------------------

flexflow_config_t
flexflow_config_create(void)
{
  FFConfig *config = new FFConfig();
  Runtime *runtime = Runtime::get_runtime();
  config->lg_hlr = runtime;
  config->lg_ctx = Runtime::get_context();
  config->field_space = runtime->create_field_space(config->lg_ctx);
  DEBUG_PRINT("[FFConfig] new %p", config);
  return FFCObjectWrapper::wrap(config);
}

void
flexflow_config_destroy(
  flexflow_config_t handle_)
{
  FFConfig *handle = FFCObjectWrapper::unwrap(handle_);
  DEBUG_PRINT("[FFConfig] delete %p", handle);
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
flexflow_config_parse_args_default(
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
  DEBUG_PRINT("[FFModel] new %p", model);
  return FFCObjectWrapper::wrap(model);
}

void
flexflow_model_destroy(
  flexflow_model_t handle_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  DEBUG_PRINT("[FFModel] delete %p", handle); 
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
flexflow_model_compile(
  flexflow_model_t handle_,
  enum LossType loss_type,
  int *metrics,
  int nb_metrics)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  std::vector<MetricsType> metrics_vec;
  for (int i = 0; i < nb_metrics; i++) {
    metrics_vec.push_back(static_cast<MetricsType>(metrics[i]));
  }
  handle->compile(loss_type, metrics_vec);
}

flexflow_tensor_t
flexflow_model_get_label_tensor(
  flexflow_model_t handle_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor *tensor = &(handle->label_tensor);
  return FFCObjectWrapper::wrap(tensor); 
}

void
flexflow_model_zero_gradients(
  flexflow_model_t handle_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  handle->zero_gradients();
}

flexflow_tensor_t
flexflow_model_add_exp(
  flexflow_model_t handle_,
  const flexflow_tensor_t x_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor *x = FFCObjectWrapper::unwrap_const(x_);
  Tensor *tensor = new Tensor();
  *tensor = handle->exp(*x);
  DEBUG_PRINT("[Exp] new Tensor %p", tensor);
  return FFCObjectWrapper::wrap(tensor);  
}
  
flexflow_tensor_t
flexflow_model_add_add(
  flexflow_model_t handle_,
  const flexflow_tensor_t x_,
  const flexflow_tensor_t y_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor *x = FFCObjectWrapper::unwrap_const(x_);
  const Tensor *y = FFCObjectWrapper::unwrap_const(y_);
  Tensor *tensor = new Tensor();
  *tensor = handle->add(*x, *y);
  DEBUG_PRINT("[Add] new Tensor %p", tensor);
  return FFCObjectWrapper::wrap(tensor);  
}
  
flexflow_tensor_t
flexflow_model_add_subtract(
  flexflow_model_t handle_,
  const flexflow_tensor_t x_,
  const flexflow_tensor_t y_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor *x = FFCObjectWrapper::unwrap_const(x_);
  const Tensor *y = FFCObjectWrapper::unwrap_const(y_);
  Tensor *tensor = new Tensor();
  *tensor = handle->subtract(*x, *y);
  DEBUG_PRINT("[Subtract] new Tensor %p", tensor);
  return FFCObjectWrapper::wrap(tensor);  
}

flexflow_tensor_t
flexflow_model_add_multiply(
  flexflow_model_t handle_,
  const flexflow_tensor_t x_,
  const flexflow_tensor_t y_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor *x = FFCObjectWrapper::unwrap_const(x_);
  const Tensor *y = FFCObjectWrapper::unwrap_const(y_);
  Tensor *tensor = new Tensor();
  *tensor = handle->multiply(*x, *y);
  DEBUG_PRINT("[Multiply] new Tensor %p", tensor);
  return FFCObjectWrapper::wrap(tensor);  
}
  
flexflow_tensor_t
flexflow_model_add_divide(
  flexflow_model_t handle_,
  const flexflow_tensor_t x_,
  const flexflow_tensor_t y_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor *x = FFCObjectWrapper::unwrap_const(x_);
  const Tensor *y = FFCObjectWrapper::unwrap_const(y_);
  Tensor *tensor = new Tensor();
  *tensor = handle->divide(*x, *y);
  DEBUG_PRINT("[Divide] new Tensor %p", tensor);
  return FFCObjectWrapper::wrap(tensor);  
}

flexflow_tensor_t
flexflow_model_add_conv2d(
  flexflow_model_t handle_,
  const flexflow_tensor_t input_,
  int out_channels,
  int kernel_h, int kernel_w,
  int stride_h, int stride_w,
  int padding_h, int padding_w,
  enum ActiMode activation /* AC_MODE_NONE */,
  bool use_bias /* True */,
  flexflow_initializer_t kernel_initializer_,
  flexflow_initializer_t bias_initializer_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor *input = FFCObjectWrapper::unwrap_const(input_);
  Tensor *tensor = new Tensor();
  Initializer *kernel_initializer = FFCObjectWrapper::unwrap(kernel_initializer_);
  Initializer *bias_initializer = FFCObjectWrapper::unwrap(bias_initializer_);
  *tensor = handle->conv2d(*input, out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, activation, use_bias, kernel_initializer, bias_initializer);
  DEBUG_PRINT("[Conv2d] new Tensor 4D %p (%d, %d, %d, %d), activation %d, use_bias %d, kernel_init %p, bias_init %p", tensor, tensor->adim[0], tensor->adim[1], tensor->adim[2], tensor->adim[3], activation, use_bias, kernel_initializer, bias_initializer);
  return FFCObjectWrapper::wrap(tensor);   
}

flexflow_op_t
flexflow_model_add_conv2d_no_inout(
  flexflow_model_t handle_,
  int in_channels,
  int out_channels,
  int kernel_h, int kernel_w,
  int stride_h, int stride_w,
  int padding_h, int padding_w,
  enum ActiMode activation /* AC_MODE_NONE */,
  bool use_bias /* True */,
  flexflow_initializer_t kernel_initializer_,
  flexflow_initializer_t bias_initializer_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Initializer *kernel_initializer = FFCObjectWrapper::unwrap(kernel_initializer_);
  Initializer *bias_initializer = FFCObjectWrapper::unwrap(bias_initializer_);
  Conv2D *conv2d = handle->conv2d(in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, activation, use_bias, kernel_initializer, bias_initializer);
  Op *op = (Op*)conv2d;
  DEBUG_PRINT("[Conv2d] no inout, layer %p, activation %d, use_bias %d, kernel_init %p, bias_init %p", conv2d, activation, use_bias, kernel_initializer, bias_initializer);
  return FFCObjectWrapper::wrap(op);   
}

flexflow_tensor_t
flexflow_model_add_embedding(
  flexflow_model_t handle_,
  const flexflow_tensor_t input_,
  int num_entires, int out_dim,
  enum AggrMode aggr,
  flexflow_initializer_t kernel_initializer_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor *input = FFCObjectWrapper::unwrap_const(input_);
  Tensor *tensor = new Tensor();
  Initializer *kernel_initializer = FFCObjectWrapper::unwrap(kernel_initializer_);
  *tensor = handle->embedding(*input, num_entires, out_dim, aggr, kernel_initializer);
  printf("[Embedding] new Tensor %p, kernel_init %p", tensor, kernel_initializer);
  return FFCObjectWrapper::wrap(tensor);   
}

flexflow_tensor_t
flexflow_model_add_pool2d(
  flexflow_model_t handle_,
  flexflow_tensor_t input_,
  int kernel_h, int kernel_w,
  int stride_h, int stride_w,
  int padding_h, int padding_w,
  enum PoolType type /* POOL_MAX */, 
  enum ActiMode activation /* AC_MODE_NONE */)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor *input = FFCObjectWrapper::unwrap(input_);
  Tensor *tensor = new Tensor();
  *tensor = handle->pool2d(*input, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, type, activation);
  DEBUG_PRINT("[Pool2d] new Tensor 4D %p (%d, %d, %d, %d), pool %d, activation %d", tensor, tensor->adim[0], tensor->adim[1], tensor->adim[2], tensor->adim[3], type, activation);
  return FFCObjectWrapper::wrap(tensor); 
}

flexflow_op_t
flexflow_model_add_pool2d_no_inout(
  flexflow_model_t handle_,
  int kernel_h, int kernel_w,
  int stride_h, int stride_w,
  int padding_h, int padding_w,
  enum PoolType type /* POOL_MAX */, 
  enum ActiMode activation /* AC_MODE_NONE */)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Pool2D *pool2d = handle->pool2d(kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, type, activation);
  Op *op = (Op*)pool2d;
  printf("[Pool2d] no inout, layer %p, pool %d, activation %d", pool2d, type, activation);
  return FFCObjectWrapper::wrap(op); 
}

flexflow_tensor_t
flexflow_model_add_batch_norm(
  flexflow_model_t handle_,
  const flexflow_tensor_t input_,
  bool relu)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor *input = FFCObjectWrapper::unwrap(input_);
  Tensor *tensor = new Tensor();
  *tensor = handle->batch_norm(*input, relu);
  DEBUG_PRINT("[BatchNorm] new Tensor 4D %p (%d, %d, %d, %d)", tensor, tensor->adim[0], tensor->adim[1], tensor->adim[2], tensor->adim[3]);
  return FFCObjectWrapper::wrap(tensor); 
}

flexflow_tensor_t
flexflow_model_add_dense(
  flexflow_model_t handle_,
  const flexflow_tensor_t input_,
  int out_dim,
  enum ActiMode activation /* AC_MODE_NONE */,
  bool use_bias /* true */,
  flexflow_initializer_t kernel_initializer_,
  flexflow_initializer_t bias_initializer_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor *input = FFCObjectWrapper::unwrap_const(input_);
  Tensor *tensor = new Tensor();
  Initializer *kernel_initializer = FFCObjectWrapper::unwrap(kernel_initializer_);
  Initializer *bias_initializer = FFCObjectWrapper::unwrap(bias_initializer_);
  *tensor = handle->dense(*input, out_dim, activation, use_bias, kernel_initializer, bias_initializer);
  DEBUG_PRINT("[Dense] new Tensor 2D %p (%d, %d, %d, %d), activation %d, use_bias %d, kernel_init %p, bias_init %p", tensor, tensor->adim[0], tensor->adim[1], tensor->adim[2], tensor->adim[3], activation, use_bias, kernel_initializer, bias_initializer);
  return FFCObjectWrapper::wrap(tensor); 
}

flexflow_op_t
flexflow_model_add_dense_no_inout(
  flexflow_model_t handle_,
  int in_dim,
  int out_dim,
  enum ActiMode activation /* AC_MODE_NONE */,
  bool use_bias /* true */,
  flexflow_initializer_t kernel_initializer_,
  flexflow_initializer_t bias_initializer_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Initializer *kernel_initializer = FFCObjectWrapper::unwrap(kernel_initializer_);
  Initializer *bias_initializer = FFCObjectWrapper::unwrap(bias_initializer_);
  Linear *linear = handle->dense(in_dim, out_dim, activation, use_bias, kernel_initializer, bias_initializer);
  DEBUG_PRINT("[Dense] no inout, layer %p, activation %d, use_bias %d, kernel_init %p, bias_init %p", linear, activation, use_bias, kernel_initializer, bias_initializer);
  return FFCObjectWrapper::wrap(linear); 
}

flexflow_tensor_t
flexflow_model_add_concat(
  flexflow_model_t handle_,
  int n,
  flexflow_tensor_t* input_,
  int axis)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor *tensor = new Tensor();
  std::vector<Tensor> input_vec;
  char cbuffer[256];
  char *cbuffer_ptr = cbuffer;
  sprintf(cbuffer_ptr, "[Concat] input tensor");
  cbuffer_ptr += 21;
  for (int i = 0; i < n; i++ ) {
    Tensor *t = FFCObjectWrapper::unwrap(input_[i]);
    input_vec.push_back(*t);
    sprintf(cbuffer_ptr, "%p ", t);
    cbuffer_ptr +=15; 
  }
  *tensor = handle->concat(n, input_vec.data(), axis);
  sprintf(cbuffer_ptr, ", concat new Tensor %p", tensor);
  DEBUG_PRINT("%s", cbuffer);
  return FFCObjectWrapper::wrap(tensor); 
}

flexflow_tensor_t
flexflow_model_add_flat(
  flexflow_model_t handle_,
  flexflow_tensor_t input_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor *input = FFCObjectWrapper::unwrap(input_);
  Tensor *tensor = new Tensor();
  *tensor = handle->flat(*input);
  DEBUG_PRINT("[Flat] new Tensor 4D %p", tensor);
  return FFCObjectWrapper::wrap(tensor);  
}

flexflow_op_t
flexflow_model_add_flat_no_inout(
  flexflow_model_t handle_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Flat *flat = handle->flat();
  Op *op = (Op*)flat;
  DEBUG_PRINT("[Flat] no inout, layer %p", flat);
  return FFCObjectWrapper::wrap(op);  
}

flexflow_tensor_t
flexflow_model_add_softmax(
  flexflow_model_t handle_,
  const flexflow_tensor_t input_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor *input = FFCObjectWrapper::unwrap(input_);
  Tensor *tensor = new Tensor();
  *tensor = handle->softmax(*input);
  DEBUG_PRINT("[Softmax] new Tensor %p", tensor);
  return FFCObjectWrapper::wrap(tensor);   
}

flexflow_tensor_t
flexflow_model_add_relu(
  flexflow_model_t handle_,
  const flexflow_tensor_t input_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor *input = FFCObjectWrapper::unwrap(input_);
  Tensor *tensor = new Tensor();
  *tensor = handle->relu(*input);
  DEBUG_PRINT("[Relu] new Tensor %p", tensor);
  return FFCObjectWrapper::wrap(tensor);   
}
  
flexflow_tensor_t
flexflow_model_add_sigmod(
  flexflow_model_t handle_,
  const flexflow_tensor_t input_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor *input = FFCObjectWrapper::unwrap(input_);
  Tensor *tensor = new Tensor();
  *tensor = handle->sigmoid(*input);
  DEBUG_PRINT("[Sigmoid] new Tensor %p", tensor);
  return FFCObjectWrapper::wrap(tensor);   
}
    
flexflow_tensor_t
flexflow_model_add_tanh(
  flexflow_model_t handle_,
  const flexflow_tensor_t input_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor *input = FFCObjectWrapper::unwrap(input_);
  Tensor *tensor = new Tensor();
  *tensor = handle->tanh(*input);
  DEBUG_PRINT("[Tanh] new Tensor %p", tensor);
  return FFCObjectWrapper::wrap(tensor);   
}
  
flexflow_tensor_t
flexflow_model_add_elu(
  flexflow_model_t handle_,
  const flexflow_tensor_t input_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor *input = FFCObjectWrapper::unwrap(input_);
  Tensor *tensor = new Tensor();
  *tensor = handle->elu(*input);
  DEBUG_PRINT("[Elu] new Tensor %p", tensor);
  return FFCObjectWrapper::wrap(tensor);   
}

flexflow_tensor_t
flexflow_model_add_dropout(
  flexflow_model_t handle_,
  const flexflow_tensor_t input_,
  float rate, 
  unsigned long long seed)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor *input = FFCObjectWrapper::unwrap(input_);
  Tensor *tensor = new Tensor();
  *tensor = handle->dropout(*input, rate, seed);
  DEBUG_PRINT("[Dropout] new Tensor %p", tensor);
  return FFCObjectWrapper::wrap(tensor);   
}

// void
// flexflow_model_add_mse_loss(
//   flexflow_model_t handle_,
//   const flexflow_tensor_t logits_,
//   const flexflow_tensor_t labels_,
//   const char* reduction)
// {
//   FFModel *handle = FFCObjectWrapper::unwrap(handle_);
//   Tensor *logits = FFCObjectWrapper::unwrap(logits_);
//   Tensor *labels = FFCObjectWrapper::unwrap(labels_);
//   handle->mse_loss(*logits, *labels, reduction);
//   DEBUG_PRINT("[MSE_Loss] reduction %s", reduction);
// }

void
flexflow_model_set_sgd_optimizer(
  flexflow_model_t handle_, 
  flexflow_sgd_optimizer_t optimizer_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  SGDOptimizer *optimizer = FFCObjectWrapper::unwrap(optimizer_);
  handle->optimizer = static_cast<Optimizer *>(optimizer);
}

void
flexflow_model_set_adam_optimizer(
  flexflow_model_t handle_, 
  flexflow_adam_optimizer_t optimizer_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  AdamOptimizer *optimizer = FFCObjectWrapper::unwrap(optimizer_);
  handle->optimizer = static_cast<Optimizer *>(optimizer);
}

void
flexflow_model_print_layers(
  flexflow_model_t handle_, 
  int id)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  handle->print_layers(id);
}

flexflow_op_t
flexflow_model_get_layer_by_id(
  flexflow_model_t handle_,
  int layer_id)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Op* layer = handle->layers[layer_id];
  return FFCObjectWrapper::wrap(layer);  
}

flexflow_parameter_t
flexflow_model_get_parameter_by_id(
  flexflow_model_t handle_,
  int layer_id)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Parameter *tensor = &(handle->parameters[layer_id]);
  return FFCObjectWrapper::wrap(tensor);  
}

flexflow_perf_metrics_t
flexflow_model_get_perf_metrics(
  flexflow_model_t handle_)
{
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  PerfMetrics *perf_metrics = new PerfMetrics();
  *perf_metrics = handle->current_metrics.get_result<PerfMetrics>();
  DEBUG_PRINT("[Model] create PerfMetrics %p, train_correct %d\n", perf_metrics, perf_metrics->train_correct);
  return FFCObjectWrapper::wrap(perf_metrics);   
}

// -----------------------------------------------------------------------
// Tensor
// -----------------------------------------------------------------------

flexflow_tensor_t
flexflow_tensor_create(
  flexflow_model_t model_,
  int num_dims,
  const int* dims,
  const char* name,
  enum DataType data_type, 
  bool create_grad /* true */)
{
  Tensor *tensor = new Tensor();
  FFModel *model = FFCObjectWrapper::unwrap(model_);
  if (num_dims == 4) {
    *tensor = model->create_tensor<4>(dims, name, data_type, create_grad);
    DEBUG_PRINT("[Tensor] new 4D %p (%d, %d, %d, %d)", tensor, tensor->adim[0], tensor->adim[1], tensor->adim[2], tensor->adim[3]);
  } else if (num_dims == 3) {
    *tensor = model->create_tensor<3>(dims, name, data_type, create_grad);
    DEBUG_PRINT("[Tensor] new 3D %p (%d, %d, %d, %d)", tensor, tensor->adim[0], tensor->adim[1], tensor->adim[2], tensor->adim[3]);
  } else if (num_dims == 2) {
    *tensor = model->create_tensor<2>(dims, name, data_type, create_grad);
    DEBUG_PRINT("[Tensor] new 2D %p (%d, %d, %d, %d)", tensor, tensor->adim[0], tensor->adim[1], tensor->adim[2], tensor->adim[3]);
  } else {
    assert(0);
  }
  return FFCObjectWrapper::wrap(tensor);
}

void
flexflow_tensor_destroy(
  flexflow_tensor_t handle_)
{
  Tensor *handle = FFCObjectWrapper::unwrap(handle_);
  DEBUG_PRINT("[Tensor] delete %p", handle);
  delete handle;
}

void
flexflow_tensor_inline_map(
  flexflow_tensor_t handle_,
  flexflow_config_t config_)
{
  Tensor *handle = FFCObjectWrapper::unwrap(handle_);
  FFConfig *config = FFCObjectWrapper::unwrap(config_);  
  handle->inline_map(*config);
}

void  
flexflow_tensor_inline_unmap(
  flexflow_tensor_t handle_,
  flexflow_config_t config_)
{
  Tensor *handle = FFCObjectWrapper::unwrap(handle_);
  FFConfig *config = FFCObjectWrapper::unwrap(config_);  
  handle->inline_unmap(*config);
}

float*  
flexflow_tensor_get_raw_ptr_float(
  flexflow_tensor_t handle_,
  flexflow_config_t config_)
{
  Tensor *handle = FFCObjectWrapper::unwrap(handle_);
  FFConfig *config = FFCObjectWrapper::unwrap(config_);  
  float *raw_ptr = handle->get_raw_ptr<float>(*config);
  return raw_ptr;
}

int32_t*  
flexflow_tensor_get_raw_ptr_int32(
  flexflow_tensor_t handle_,
  flexflow_config_t config_)
{
  Tensor *handle = FFCObjectWrapper::unwrap(handle_);
  FFConfig *config = FFCObjectWrapper::unwrap(config_);  
  int32_t *raw_ptr = handle->get_raw_ptr<int32_t>(*config);
  return raw_ptr;
}

int
flexflow_tensor_get_num_dims(
  flexflow_tensor_t handle_)
{
  Tensor *handle = FFCObjectWrapper::unwrap(handle_);
  return handle->numDim;
}

int*
flexflow_tensor_get_dims(
  flexflow_tensor_t handle_)
{
  Tensor *handle = FFCObjectWrapper::unwrap(handle_);
  DEBUG_PRINT("[Tensor] get dims [%d, %d, %d, %d]", handle->adim[3], handle->adim[2], handle->adim[1], handle->adim[0]);
  return &(handle->adim[0]);
}

int
flexflow_tensor_get_data_type(
  flexflow_tensor_t handle_)
{
  Tensor *handle = FFCObjectWrapper::unwrap(handle_);
  return static_cast<int>(handle->data_type);
}

void
flexflow_tensor_attach_raw_ptr(
  flexflow_tensor_t handle_,
  flexflow_config_t config_,
  void *raw_ptr,
  bool column_major)
{
  Tensor *handle = FFCObjectWrapper::unwrap(handle_);
  FFConfig *config = FFCObjectWrapper::unwrap(config_);  
  handle->attach_raw_ptr(*config, raw_ptr, column_major);  
  DEBUG_PRINT("[Tensor] attach numpy array: ptr %p, column_major %d", raw_ptr, column_major);
}

void
flexflow_tensor_detach_raw_ptr(
  flexflow_tensor_t handle_,
  flexflow_config_t config_)
{
  Tensor *handle = FFCObjectWrapper::unwrap(handle_);
  FFConfig *config = FFCObjectWrapper::unwrap(config_);
  handle->detach_raw_ptr(*config);  
}

bool
flexflow_tensor_is_mapped(
  flexflow_tensor_t handle_)
{
  Tensor *handle = FFCObjectWrapper::unwrap(handle_);
  return handle->physical_region.is_mapped();
}

// -----------------------------------------------------------------------
// Parameter
// -----------------------------------------------------------------------

bool
flexflow_parameter_set_weights_float(
  flexflow_parameter_t handle_,
  flexflow_model_t model_,
  int num_dim,
  int *dims,
  const float *data)
{
  Parameter *handle = FFCObjectWrapper::unwrap(handle_);
  const FFModel *model = FFCObjectWrapper::unwrap_const(model_);
  std::vector<int> dims_vec;
  for (int i = 0; i < num_dim; i++ ) {
    dims_vec.push_back(dims[i]);
  }
  return handle->set_weights<float>(*model, dims_vec, data);
}

bool
flexflow_parameter_get_weights_float(
  flexflow_parameter_t handle_,
  flexflow_model_t model_,
  float *data)
{
  Parameter *handle = FFCObjectWrapper::unwrap(handle_);
  const FFModel *model = FFCObjectWrapper::unwrap_const(model_);
  return handle->get_weights<float>(*model, data);
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
  DEBUG_PRINT("[SGDOptimizer] new %p", optimizer);
  return FFCObjectWrapper::wrap(optimizer);
}

void 
flexflow_sgd_optimizer_destroy(
  flexflow_sgd_optimizer_t handle_)
{
  SGDOptimizer *handle = FFCObjectWrapper::unwrap(handle_);
  DEBUG_PRINT("[SGDOptimizer] delete %p", handle);
  delete handle;
}

void 
flexflow_sgd_optimizer_set_lr(
  flexflow_sgd_optimizer_t handle_, 
  double lr)
{
  SGDOptimizer *handle = FFCObjectWrapper::unwrap(handle_);
  handle->lr = lr;
}

// -----------------------------------------------------------------------
// AdamOptimizer
// -----------------------------------------------------------------------

flexflow_adam_optimizer_t
flexflow_adam_optimizer_create(
  flexflow_model_t model_,
  double alpha /*0.001f*/, 
  double beta1 /*0.9f*/,
  double beta2 /*0.999f*/, 
  double weight_decay /*0.0f*/,
  double epsilon /*1e-8*/)
{
  const FFModel *model = FFCObjectWrapper::unwrap_const(model_);
  AdamOptimizer *optimizer = new AdamOptimizer(model, alpha, beta1, beta2, weight_decay, epsilon);
  DEBUG_PRINT("AdamOptimizer new %p", optimizer);
  return FFCObjectWrapper::wrap(optimizer);
}

void 
flexflow_adam_optimizer_destroy(
  flexflow_adam_optimizer_t handle_)
{
  AdamOptimizer *handle = FFCObjectWrapper::unwrap(handle_);
  DEBUG_PRINT("AdamOptimizer delete %p", handle);
  delete handle;
}

void 
flexflow_adam_optimizer_set_lr(
  flexflow_adam_optimizer_t handle_, 
  double lr)
{
  AdamOptimizer *handle = FFCObjectWrapper::unwrap(handle_);
  handle->alpha = lr;
}

// -----------------------------------------------------------------------
// Initializer
// -----------------------------------------------------------------------
flexflow_initializer_t
flexflow_initializer_create_null()
{
  Initializer *initializer = NULL;
  return FFCObjectWrapper::wrap(initializer); 
}

// -----------------------------------------------------------------------
// GlorotUniform
// -----------------------------------------------------------------------

flexflow_glorot_uniform_initializer_t
flexflow_glorot_uniform_initializer_create(
  int seed)
{
  GlorotUniform *initializer = new GlorotUniform(seed);
  DEBUG_PRINT("[GlorotUniform] new %p", initializer);
  return FFCObjectWrapper::wrap(initializer); 
}

void  
flexflow_glorot_uniform_initializer_destroy(
  flexflow_glorot_uniform_initializer_t handle_)
{
  GlorotUniform *handle = FFCObjectWrapper::unwrap(handle_);
  DEBUG_PRINT("[GlorotUniform] delete %p", handle);
  delete handle;
}

// -----------------------------------------------------------------------
// ZeroInitializer
// -----------------------------------------------------------------------

flexflow_zero_initializer_t
flexflow_zero_initializer_create(void)
{
  ZeroInitializer *initializer = new ZeroInitializer();
  DEBUG_PRINT("[ZeroInitializer] new %p", initializer);
  return FFCObjectWrapper::wrap(initializer); 
}

void  
flexflow_zero_initializer_destroy(
  flexflow_zero_initializer_t handle_)
{
  ZeroInitializer *handle = FFCObjectWrapper::unwrap(handle_);
  DEBUG_PRINT("[ZeroInitializer] delete %p", handle);
  delete handle;
}

// -----------------------------------------------------------------------
// UniformInitializer
// -----------------------------------------------------------------------

flexflow_uniform_initializer_t
flexflow_uniform_initializer_create(
  int seed, 
  float min, 
  float max)
{
  UniformInitializer *initializer = new UniformInitializer(seed, min, max);
  DEBUG_PRINT("[UniformInitializer] new %p", initializer);
  return FFCObjectWrapper::wrap(initializer);  
}

void  
flexflow_uniform_initializer_destroy(
  flexflow_uniform_initializer_t handle_)
{
  UniformInitializer *handle = FFCObjectWrapper::unwrap(handle_);
  DEBUG_PRINT("[UniformInitializer] delete %p", handle);
  delete handle;
}

// -----------------------------------------------------------------------
// NormInitializer
// -----------------------------------------------------------------------

flexflow_norm_initializer_t
flexflow_norm_initializer_create(
  int seed, 
  float mean, 
  float stddev)
{
  NormInitializer *initializer = new NormInitializer(seed, mean, stddev);
  DEBUG_PRINT("[NormInitializer] new %p", initializer);
  return FFCObjectWrapper::wrap(initializer);  
}

void  
flexflow_norm_initializer_destroy(
  flexflow_norm_initializer_t handle_)
{
  NormInitializer *handle = FFCObjectWrapper::unwrap(handle_);
  DEBUG_PRINT("[NormInitializer] delete %p", handle);
  delete handle;
}

// -----------------------------------------------------------------------
// PerfMetrics
// -----------------------------------------------------------------------
void
flexflow_per_metrics_destroy(
  flexflow_perf_metrics_t handle_)
{
  PerfMetrics *handle = FFCObjectWrapper::unwrap(handle_);
  delete handle;
  DEBUG_PRINT("[PerfMetrics] delete PerfMetrics %p", handle);
}

float
flexflow_per_metrics_get_accuracy(
  flexflow_perf_metrics_t handle_)
{
  PerfMetrics *handle = FFCObjectWrapper::unwrap(handle_);
  float accuracy = handle->train_correct * 100.0f / handle->train_all;
  return accuracy;
}

// -----------------------------------------------------------------------
// NetConfig
// -----------------------------------------------------------------------
flexflow_net_config_t
flexflow_net_config_create()
{
  NetConfig *netconfig = new NetConfig();
  return FFCObjectWrapper::wrap(netconfig);  
}

void
flexflow_net_config_destroy(
  flexflow_net_config_t handle_)
{
  NetConfig *handle = FFCObjectWrapper::unwrap(handle_);
  delete handle;
}

const char*
flexflow_net_config_get_dataset_path(
  flexflow_net_config_t handle_)
{
  NetConfig *handle = FFCObjectWrapper::unwrap(handle_);
  const char *cstr = handle->dataset_path.c_str();
  return cstr;
}

// -----------------------------------------------------------------------
// DataLoader
// -----------------------------------------------------------------------

flexflow_dataloader_4d_t
flexflow_dataloader_4d_create(
  flexflow_model_t ffmodel_, 
  flexflow_net_config_t netconfig_,
  flexflow_tensor_t input_, 
  flexflow_tensor_t label_)
{
  FFModel *ffmodel = FFCObjectWrapper::unwrap(ffmodel_);
  NetConfig *netconfig = FFCObjectWrapper::unwrap(netconfig_);
  Tensor *input = FFCObjectWrapper::unwrap(input_);
  Tensor *label = FFCObjectWrapper::unwrap(label_);
  ImgDataLoader4D *dataloader = new ImgDataLoader4D(*ffmodel, *netconfig, *input, *label);
  return FFCObjectWrapper::wrap(dataloader);  
}

flexflow_dataloader_4d_t
flexflow_dataloader_4d_create_v2(
  flexflow_model_t ffmodel_, 
  flexflow_tensor_t input_, 
  flexflow_tensor_t label_,
  flexflow_tensor_t full_input_, 
  flexflow_tensor_t full_label_,
  int num_samples)
{
  FFModel *ffmodel = FFCObjectWrapper::unwrap(ffmodel_);
  Tensor *input = FFCObjectWrapper::unwrap(input_);
  Tensor *label = FFCObjectWrapper::unwrap(label_);
  Tensor *full_input = FFCObjectWrapper::unwrap(full_input_);
  Tensor *full_label = FFCObjectWrapper::unwrap(full_label_);
  ImgDataLoader4D *dataloader = new ImgDataLoader4D(*ffmodel, *input, *label, *full_input, *full_label, num_samples);
  return FFCObjectWrapper::wrap(dataloader);  
}

void  
flexflow_dataloader_4d_destroy(
  flexflow_dataloader_4d_t handle_)
{
  ImgDataLoader *handle = FFCObjectWrapper::unwrap(handle_);
  delete handle;
}

void
flexflow_dataloader_4d_set_num_samples(
  flexflow_dataloader_4d_t handle_,
  int samples)
{
  ImgDataLoader4D *handle = FFCObjectWrapper::unwrap(handle_);
  handle->num_samples = samples;  
  DEBUG_PRINT("[Dataloader4D] set number of samples %d", samples);
}

int
flexflow_dataloader_4d_get_num_samples(
  flexflow_dataloader_4d_t handle_)
{
  ImgDataLoader4D *handle = FFCObjectWrapper::unwrap(handle_);
  return handle->num_samples;
}

void
flexflow_dataloader_4d_reset(
  flexflow_dataloader_4d_t handle_)
{
  ImgDataLoader4D *handle = FFCObjectWrapper::unwrap(handle_);
  handle->reset();
}

void
flowflow_dataloader_4d_next_batch(
  flexflow_dataloader_4d_t handle_,
  flexflow_model_t ffmodel_)
{
  ImgDataLoader4D *handle = FFCObjectWrapper::unwrap(handle_);
  FFModel *ffmodel = FFCObjectWrapper::unwrap(ffmodel_);
  handle->next_batch(*ffmodel);
}

flexflow_dataloader_2d_t
flexflow_dataloader_2d_create_v2(
  flexflow_model_t ffmodel_, 
  flexflow_tensor_t input_, 
  flexflow_tensor_t label_,
  flexflow_tensor_t full_input_, 
  flexflow_tensor_t full_label_,
  int num_samples)
{
  FFModel *ffmodel = FFCObjectWrapper::unwrap(ffmodel_);
  Tensor *input = FFCObjectWrapper::unwrap(input_);
  Tensor *label = FFCObjectWrapper::unwrap(label_);
  Tensor *full_input = FFCObjectWrapper::unwrap(full_input_);
  Tensor *full_label = FFCObjectWrapper::unwrap(full_label_);
  ImgDataLoader2D *dataloader = new ImgDataLoader2D(*ffmodel, *input, *label, *full_input, *full_label, num_samples);
  return FFCObjectWrapper::wrap(dataloader);  
}

void  
flexflow_dataloader_2d_destroy(
  flexflow_dataloader_2d_t handle_)
{
  ImgDataLoader2D *handle = FFCObjectWrapper::unwrap(handle_);
  delete handle;
}

void
flexflow_dataloader_2d_set_num_samples(
  flexflow_dataloader_2d_t handle_,
  int samples)
{
  ImgDataLoader2D *handle = FFCObjectWrapper::unwrap(handle_);
  handle->num_samples = samples;  
  DEBUG_PRINT("[Dataloader2D] set number of samples %d", samples);
}

int
flexflow_dataloader_2d_get_num_samples(
  flexflow_dataloader_2d_t handle_)
{
  ImgDataLoader2D *handle = FFCObjectWrapper::unwrap(handle_);
  return handle->num_samples;
}

void
flexflow_dataloader_2d_reset(
  flexflow_dataloader_2d_t handle_)
{
  ImgDataLoader2D *handle = FFCObjectWrapper::unwrap(handle_);
  handle->reset();
}

void
flowflow_dataloader_2d_next_batch(
  flexflow_dataloader_2d_t handle_,
  flexflow_model_t ffmodel_)
{
  ImgDataLoader2D *handle = FFCObjectWrapper::unwrap(handle_);
  FFModel *ffmodel = FFCObjectWrapper::unwrap(ffmodel_);
  handle->next_batch(*ffmodel);
}

// -----------------------------------------------------------------------
// Single Dataloader
// -----------------------------------------------------------------------

flexflow_single_dataloader_t
flexflow_single_dataloader_create(
  flexflow_model_t ffmodel_, 
  flexflow_tensor_t input_, 
  flexflow_tensor_t full_input_, 
  int num_samples,
  enum DataType data_type)
{
  FFModel *ffmodel = FFCObjectWrapper::unwrap(ffmodel_);
  Tensor *input = FFCObjectWrapper::unwrap(input_);
  Tensor *full_input = FFCObjectWrapper::unwrap(full_input_);
  SingleDataLoader *dataloader = new SingleDataLoader(*ffmodel, *input, *full_input, num_samples, data_type);
  return FFCObjectWrapper::wrap(dataloader);  
}

void  
flexflow_single_dataloader_destroy(
  flexflow_single_dataloader_t handle_)
{
  SingleDataLoader *handle = FFCObjectWrapper::unwrap(handle_);
  DEBUG_PRINT("[SingleDataLoader] delete %p", handle);
  delete handle;
}

void
flexflow_single_dataloader_set_num_samples(
  flexflow_single_dataloader_t handle_,
  int samples)
{
  SingleDataLoader *handle = FFCObjectWrapper::unwrap(handle_);
  handle->num_samples = samples;  
  DEBUG_PRINT("[SingleDataloader] set number of samples %d", samples);
}

int
flexflow_single_dataloader_get_num_samples(
  flexflow_single_dataloader_t handle_)
{
  SingleDataLoader *handle = FFCObjectWrapper::unwrap(handle_);
  return handle->num_samples;
}

void
flexflow_single_dataloader_reset(
  flexflow_single_dataloader_t handle_)
{
  SingleDataLoader *handle = FFCObjectWrapper::unwrap(handle_);
  handle->reset();
}

void
flowflow_single_dataloader_next_batch(
  flexflow_single_dataloader_t handle_,
  flexflow_model_t ffmodel_)
{
  SingleDataLoader *handle = FFCObjectWrapper::unwrap(handle_);
  FFModel *ffmodel = FFCObjectWrapper::unwrap(ffmodel_);
  handle->next_batch(*ffmodel);
}

// -----------------------------------------------------------------------
// Timer
// -----------------------------------------------------------------------

double
flexflow_get_current_time(
  flexflow_config_t config_)
{
  FFConfig *config = FFCObjectWrapper::unwrap(config_);
  config->lg_hlr->issue_execution_fence(config->lg_ctx);
  TimingLauncher timer(MEASURE_MICRO_SECONDS);
  Future future = config->lg_hlr->issue_timing_measurement(config->lg_ctx, timer);
  future.get_void_result();
  double ts_start = Realm::Clock::current_time_in_microseconds();
  return ts_start;
}

// -----------------------------------------------------------------------
// Trace
// -----------------------------------------------------------------------

void
flexflow_begin_trace(
  flexflow_config_t config_, 
  int trace_id)
{
  FFConfig *config = FFCObjectWrapper::unwrap(config_);
  config->lg_hlr->begin_trace(config->lg_ctx, trace_id);
}

void
flexflow_end_trace(
  flexflow_config_t config_, 
  int trace_id)
{
  FFConfig *config = FFCObjectWrapper::unwrap(config_);
  config->lg_hlr->end_trace(config->lg_ctx, trace_id);
}

// -----------------------------------------------------------------------
// Op
// -----------------------------------------------------------------------

flexflow_parameter_t
flexflow_op_get_parameter_by_id(
  flexflow_op_t handle_,
  int id)
{
  Op *handle = FFCObjectWrapper::unwrap(handle_);
  Parameter *tensor = handle->get_parameter(id);
  return FFCObjectWrapper::wrap(tensor);  
}

flexflow_tensor_t
flexflow_op_get_input_by_id(
  flexflow_op_t handle_,
  int id)
{
  Op *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor *tensor = &(handle->inputs[id]);
  return FFCObjectWrapper::wrap(tensor); 
}

flexflow_tensor_t
flexflow_op_get_output_by_id(
  flexflow_op_t handle_,
  int id)
{
  Op *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor *tensor = &(handle->outputs[id]);
  return FFCObjectWrapper::wrap(tensor);     
}

void
flexflow_op_init(
  flexflow_op_t handle_,
  flexflow_model_t model_)
{
  Op *handle = FFCObjectWrapper::unwrap(handle_);
  FFModel *model = FFCObjectWrapper::unwrap(model_);
  handle->init(*model);
} 

flexflow_tensor_t
flexflow_op_init_inout(
  flexflow_op_t handle_,
  flexflow_model_t model_,
  flexflow_tensor_t input_)
{
  Op *handle = FFCObjectWrapper::unwrap(handle_);
  FFModel *model = FFCObjectWrapper::unwrap(model_);
  Tensor *input = FFCObjectWrapper::unwrap(input_);
  Tensor *tensor = new Tensor();
  *tensor = handle->init_inout(*model, *input);
  DEBUG_PRINT("[Op] init inout new Tensor %p", tensor);
  return FFCObjectWrapper::wrap(tensor);   
}

void
flexflow_op_forward(
  flexflow_op_t handle_,
  flexflow_model_t model_)
{
  Op *handle = FFCObjectWrapper::unwrap(handle_);
  FFModel *model = FFCObjectWrapper::unwrap(model_);
  handle->forward(*model);
}

void
flexflow_op_add_to_model(
  flexflow_op_t handle_,
  flexflow_model_t model_)
{
  // Op *handle = FFCObjectWrapper::unwrap(handle_);
  // FFModel *model = FFCObjectWrapper::unwrap(model_);
  assert(false);
  // Temporarily remove the following API
  // add an assertion in case anyone uses this API
  //handle->add_to_model(*model);
}

// -----------------------------------------------------------------------
// NetConfig implementation
// -----------------------------------------------------------------------
NetConfig::NetConfig(void)
{
  const InputArgs &command_args = Runtime::get_input_args();
  char **argv = command_args.argv;
  int argc = command_args.argc;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--dataset")) {
      dataset_path = std::string(argv[++i]);
      continue;
    }
  }
}

void register_c_custom_tasks()
{
  // 4D Load entire dataset
  {
    TaskVariantRegistrar registrar(CUSTOM_CPU_TASK_ID_1, "4D Load Entire Dataset");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<ImgDataLoader4D::load_entire_dataset>(
        registrar, "4D Load Entire Dataset Task");
  }
  // 4D Load entire dataset from numpy
  {
    TaskVariantRegistrar registrar(CUSTOM_CPU_TASK_ID_2, "4D Load Entire Dataset Numpy");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<ImgDataLoader4D::load_entire_dataset_from_numpy>(
        registrar, "4D Load Entire Dataset Task Numpy");
  }
  // 2D Load entire dataset from numpy
  {
    TaskVariantRegistrar registrar(CUSTOM_CPU_TASK_ID_3, "2D Load Entire Dataset Numpy");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<ImgDataLoader2D::load_entire_dataset_from_numpy>(
        registrar, "2D Load Entire Dataset Task Numpy");
  }
  // 4D Load input
  {
    TaskVariantRegistrar registrar(CUSTOM_GPU_TASK_ID_1, "4D Load Inputs");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<ImgDataLoader4D::load_input>(
        registrar, "4D Load Input Task");
  }
  // Load label
  {
    TaskVariantRegistrar registrar(CUSTOM_GPU_TASK_ID_2, "Load Labels");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<ImgDataLoader::load_label>(
        registrar, "Load Label Task");
  }
  // 2D Load input
  {
    TaskVariantRegistrar registrar(CUSTOM_GPU_TASK_ID_3, "2D Load Inputs");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<ImgDataLoader2D::load_input>(
        registrar, "2D Load Input Task");
  }

  SingleDataLoader::register_cpu_tasks();
  
  SingleDataLoader::register_gpu_tasks();
}
