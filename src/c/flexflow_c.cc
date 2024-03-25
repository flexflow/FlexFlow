/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
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

#include "flexflow/flexflow_c.h"
#include "flexflow/dataloader.h"
#include "flexflow/mapper.h"
#include "flexflow/request_manager.h"
#include "flexflow/utils/file_loader.h"

using namespace Legion;
using namespace FlexFlow;

class FFCObjectWrapper {
public:
#define FF_NEW_OPAQUE_WRAPPER(T_, T)                                           \
  static T_ wrap(T t) {                                                        \
    T_ t_;                                                                     \
    t_.impl = static_cast<void *>(t);                                          \
    return t_;                                                                 \
  }                                                                            \
  static const T_ wrap_const(const T t) {                                      \
    T_ t_;                                                                     \
    t_.impl = const_cast<void *>(static_cast<void const *>(t));                \
    return t_;                                                                 \
  }                                                                            \
  static T unwrap(T_ t_) { return static_cast<T>(t_.impl); }                   \
  static const T unwrap_const(const T_ t_) {                                   \
    return static_cast<const T>(t_.impl);                                      \
  }

  FF_NEW_OPAQUE_WRAPPER(flexflow_config_t, FFConfig *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_model_t, FFModel *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_tensor_t, Tensor);
  FF_NEW_OPAQUE_WRAPPER(flexflow_parallel_tensor_t, ParallelTensor);
  FF_NEW_OPAQUE_WRAPPER(flexflow_sgd_optimizer_t, SGDOptimizer *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_adam_optimizer_t, AdamOptimizer *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_initializer_t, Initializer *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_glorot_uniform_initializer_t, GlorotUniform *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_zero_initializer_t, ZeroInitializer *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_uniform_initializer_t, UniformInitializer *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_norm_initializer_t, NormInitializer *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_op_t, Layer *);
  // FF_NEW_OPAQUE_WRAPPER(flexflow_parameter_t, Parameter *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_perf_metrics_t, PerfMetrics *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_net_config_t, NetConfig *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_dlrm_config_t, DLRMConfig *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_single_dataloader_t, SingleDataLoader *);
  // inference
  FF_NEW_OPAQUE_WRAPPER(flexflow_batch_config_t, BatchConfig *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_tree_verify_batch_config_t,
                        TreeVerifyBatchConfig *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_beam_search_batch_config_t,
                        BeamSearchBatchConfig *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_inference_manager_t, InferenceManager *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_request_manager_t, RequestManager *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_file_data_loader_t, FileDataLoader *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_generation_result_t, GenerationResult *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_lora_linear_config_t, LoraLinearConfig *);
  FF_NEW_OPAQUE_WRAPPER(flexflow_peft_model_id_t, PEFTModelID *);
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

flexflow_config_t flexflow_config_create(void) {
  FFConfig *config = new FFConfig();
  DEBUG_PRINT("[FFConfig] new %p", config);
  return FFCObjectWrapper::wrap(config);
}

void flexflow_config_destroy(flexflow_config_t handle_) {
  FFConfig *handle = FFCObjectWrapper::unwrap(handle_);
  DEBUG_PRINT("[FFConfig] delete %p", handle);
  delete handle;
}

void flexflow_config_parse_args(flexflow_config_t handle_,
                                char **argv,
                                int argc) {
  FFConfig *handle = FFCObjectWrapper::unwrap(handle_);
  handle->parse_args(argv, argc);
}

void flexflow_config_parse_args_default(flexflow_config_t handle_) {
  FFConfig *handle = FFCObjectWrapper::unwrap(handle_);
  InputArgs const &command_args = Runtime::get_input_args();
  char **argv = command_args.argv;
  int argc = command_args.argc;
  handle->parse_args(argv, argc);
}

int flexflow_config_get_batch_size(flexflow_config_t handle_) {
  FFConfig *handle = FFCObjectWrapper::unwrap(handle_);
  return handle->batchSize;
}

int flexflow_config_get_workers_per_node(flexflow_config_t handle_) {
  FFConfig *handle = FFCObjectWrapper::unwrap(handle_);
  return handle->workersPerNode;
}

int flexflow_config_get_num_nodes(flexflow_config_t handle_) {
  FFConfig *handle = FFCObjectWrapper::unwrap(handle_);
  return handle->numNodes;
}

int flexflow_config_get_epochs(flexflow_config_t handle_) {
  FFConfig *handle = FFCObjectWrapper::unwrap(handle_);
  return handle->epochs;
}

bool flexflow_config_get_enable_control_replication(flexflow_config_t handle_) {
  FFConfig *handle = FFCObjectWrapper::unwrap(handle_);
  return handle->enable_control_replication;
}

int flexflow_config_get_data_parallelism_degree(flexflow_config_t handle_) {
  FFConfig *handle = FFCObjectWrapper::unwrap(handle_);
  return handle->data_parallelism_degree;
}

int flexflow_config_get_tensor_parallelism_degree(flexflow_config_t handle_) {
  FFConfig *handle = FFCObjectWrapper::unwrap(handle_);
  return handle->tensor_parallelism_degree;
}

int flexflow_config_get_pipeline_parallelism_degree(flexflow_config_t handle_) {
  FFConfig *handle = FFCObjectWrapper::unwrap(handle_);
  return handle->pipeline_parallelism_degree;
}

void flexflow_config_set_data_parallelism_degree(flexflow_config_t handle_,
                                                 int value) {
  FFConfig *handle = FFCObjectWrapper::unwrap(handle_);
  handle->data_parallelism_degree = value;
}

void flexflow_config_set_tensor_parallelism_degree(flexflow_config_t handle_,
                                                   int value) {
  FFConfig *handle = FFCObjectWrapper::unwrap(handle_);
  handle->tensor_parallelism_degree = value;
}

void flexflow_config_set_pipeline_parallelism_degree(flexflow_config_t handle_,
                                                     int value) {
  FFConfig *handle = FFCObjectWrapper::unwrap(handle_);
  handle->pipeline_parallelism_degree = value;
}

int flexflow_config_get_python_data_loader_type(flexflow_config_t handle_) {
  FFConfig *handle = FFCObjectWrapper::unwrap(handle_);
  return handle->python_data_loader_type;
}
bool flexflow_config_get_offload(flexflow_config_t handle_) {
  FFConfig *handle = FFCObjectWrapper::unwrap(handle_);
  return handle->cpu_offload;
}

// -----------------------------------------------------------------------
// FFModel
// -----------------------------------------------------------------------

flexflow_model_t flexflow_model_create(flexflow_config_t config_,
                                       bool cpu_offload) {
  FFConfig *config = FFCObjectWrapper::unwrap(config_);
  FFModel *model = new FFModel(*config, cpu_offload);
  DEBUG_PRINT("[FFModel] new %p", model);
  return FFCObjectWrapper::wrap(model);
}

void flexflow_model_destroy(flexflow_model_t handle_) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  DEBUG_PRINT("[FFModel] delete %p", handle);
  delete handle;
}

void flexflow_model_reset_metrics(flexflow_model_t handle_) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  handle->reset_metrics();
}

void flexflow_model_init_layers(flexflow_model_t handle_) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  handle->init_operators();
}

void flexflow_model_prefetch(flexflow_model_t handle_) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  handle->prefetch();
}

void flexflow_model_forward(flexflow_model_t handle_, int seq_length) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  handle->forward(seq_length);
}

void flexflow_model_backward(flexflow_model_t handle_, int seq_length) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  handle->backward(seq_length);
}

void flexflow_model_compute_metrics(flexflow_model_t handle_) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  handle->compute_metrics();
}

void flexflow_model_update(flexflow_model_t handle_) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  handle->update();
}

void flexflow_model_compile(flexflow_model_t handle_,
                            enum LossType loss_type,
                            int *metrics,
                            int nb_metrics,
                            enum CompMode comp_mode) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  std::vector<MetricsType> metrics_vec;
  for (int i = 0; i < nb_metrics; i++) {
    metrics_vec.push_back(static_cast<MetricsType>(metrics[i]));
  }
  handle->compile(loss_type, metrics_vec, comp_mode);
}

flexflow_tensor_t flexflow_model_get_label_tensor(flexflow_model_t handle_) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor tensor = handle->label_tensor;
  return FFCObjectWrapper::wrap(tensor);
}

void flexflow_model_zero_gradients(flexflow_model_t handle_) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  handle->zero_gradients();
}

flexflow_tensor_t flexflow_model_add_exp(flexflow_model_t handle_,
                                         const flexflow_tensor_t x_,
                                         char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor x = FFCObjectWrapper::unwrap_const(x_);
  Tensor tensor = handle->exp(x, name);
  DEBUG_PRINT("[Exp] new Tensor %p, x %p, name %s", tensor, x, name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_sin(flexflow_model_t handle_,
                                         const flexflow_tensor_t x_,
                                         char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor x = FFCObjectWrapper::unwrap_const(x_);
  Tensor tensor = handle->sin(x, name);
  DEBUG_PRINT("[Sin] new Tensor %p, x %p, name %s", tensor, x, name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_cos(flexflow_model_t handle_,
                                         const flexflow_tensor_t x_,
                                         char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor x = FFCObjectWrapper::unwrap_const(x_);
  Tensor tensor = handle->cos(x, name);
  DEBUG_PRINT("[Cos] new Tensor %p, x %p, name %s", tensor, x, name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_add(flexflow_model_t handle_,
                                         const flexflow_tensor_t x_,
                                         const flexflow_tensor_t y_,
                                         bool inplace_a,
                                         char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor x = FFCObjectWrapper::unwrap_const(x_);
  const Tensor y = FFCObjectWrapper::unwrap_const(y_);
  Tensor tensor = handle->add(x, y, inplace_a, name);
  DEBUG_PRINT("[Add] new Tensor %p, x %p, y %p, name %s", tensor, x, y, name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_subtract(flexflow_model_t handle_,
                                              const flexflow_tensor_t x_,
                                              const flexflow_tensor_t y_,
                                              bool inplace_a,
                                              char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor x = FFCObjectWrapper::unwrap_const(x_);
  const Tensor y = FFCObjectWrapper::unwrap_const(y_);
  Tensor tensor = handle->subtract(x, y, inplace_a, name);
  DEBUG_PRINT(
      "[Subtract] new Tensor %p, x %p, y %p, name %s", tensor, x, y, name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_multiply(flexflow_model_t handle_,
                                              const flexflow_tensor_t x_,
                                              const flexflow_tensor_t y_,
                                              bool inplace_a,
                                              char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor x = FFCObjectWrapper::unwrap_const(x_);
  const Tensor y = FFCObjectWrapper::unwrap_const(y_);
  Tensor tensor = handle->multiply(x, y, inplace_a, name);
  DEBUG_PRINT(
      "[Multiply] new Tensor %p, x %p, y %p, name %s", tensor, x, y, name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_divide(flexflow_model_t handle_,
                                            const flexflow_tensor_t x_,
                                            const flexflow_tensor_t y_,
                                            bool inplace_a,
                                            char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor x = FFCObjectWrapper::unwrap_const(x_);
  const Tensor y = FFCObjectWrapper::unwrap_const(y_);
  Tensor tensor = handle->divide(x, y, inplace_a, name);
  DEBUG_PRINT(
      "[Divide] new Tensor %p, x %p, y %p, name %s", tensor, x, y, name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_max(flexflow_model_t handle_,
                                         const flexflow_tensor_t x_,
                                         const flexflow_tensor_t y_,
                                         bool inplace_a,
                                         char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor x = FFCObjectWrapper::unwrap_const(x_);
  const Tensor y = FFCObjectWrapper::unwrap_const(y_);
  Tensor tensor = handle->max(x, y, inplace_a, name);
  DEBUG_PRINT("[Max] new Tensor %p, x %p, y %p, name %s", tensor, x, y, name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_min(flexflow_model_t handle_,
                                         const flexflow_tensor_t x_,
                                         const flexflow_tensor_t y_,
                                         bool inplace_a,
                                         char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor x = FFCObjectWrapper::unwrap_const(x_);
  const Tensor y = FFCObjectWrapper::unwrap_const(y_);
  Tensor tensor = handle->min(x, y, inplace_a, name);
  DEBUG_PRINT("[Min] new Tensor %p, x %p, y %p, name %s", tensor, x, y, name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_reduce_sum(flexflow_model_t handle_,
                                                const flexflow_tensor_t input_,
                                                int *axes,
                                                int n,
                                                bool keepdims,
                                                char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor input = FFCObjectWrapper::unwrap(input_);
  std::vector<int> axes_vec;
  for (int i = 0; i < n; i++) {
    axes_vec.push_back(axes[i]);
  }
  Tensor tensor = handle->reduce_sum(input, axes_vec, keepdims, name);
  DEBUG_PRINT("[ReduceSum] new Tensor %p, input %p, n %d, name %s",
              tensor,
              input,
              n,
              name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_rsqrt(flexflow_model_t handle_,
                                           const flexflow_tensor_t input_,
                                           char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor input = FFCObjectWrapper::unwrap(input_);
  Tensor tensor = handle->rsqrt(input, name);
  DEBUG_PRINT("[Rsqrt] new Tensor %p, input %p, name %s", tensor, input, name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_pow(flexflow_model_t handle_,
                                         const flexflow_tensor_t input_,
                                         float const exponent,
                                         char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor input = FFCObjectWrapper::unwrap(input_);
  Tensor tensor = handle->pow(input, exponent, name);
  DEBUG_PRINT("[Pow] new Tensor %p, input %p, exponent %f, name %s",
              tensor,
              input,
              exponent,
              name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_mean(flexflow_model_t handle_,
                                          const flexflow_tensor_t input_,
                                          int *dims,
                                          int n,
                                          bool keepdims,
                                          char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor input = FFCObjectWrapper::unwrap(input_);
  std::vector<int> dims_vec;
  char cbuffer[256];
  char *cbuffer_ptr = cbuffer;
  snprintf(cbuffer_ptr, 13, "[Mean] dims ");
  cbuffer_ptr += 12;
  for (int i = 0; i < n; ++i) {
    int dim = dims[i];
    dims_vec.push_back(dim);
    std::string dim_str = std::to_string(dim);
    size_t num_digits = dim_str.size();
    snprintf(cbuffer_ptr, num_digits + 2, "%s ", dim_str.c_str());
    cbuffer_ptr += num_digits + 1;
  }
  Tensor tensor = handle->mean(input, dims_vec, keepdims, name);
  DEBUG_PRINT("%s, new Tensor %p, keepdims %d, name %s",
              cbuffer,
              tensor,
              keepdims,
              name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t
    flexflow_model_add_conv2d(flexflow_model_t handle_,
                              const flexflow_tensor_t input_,
                              int out_channels,
                              int kernel_h,
                              int kernel_w,
                              int stride_h,
                              int stride_w,
                              int padding_h,
                              int padding_w,
                              enum ActiMode activation /* AC_MODE_NONE */,
                              int groups,
                              bool use_bias /* True */,
                              flexflow_op_t shared_op_,
                              flexflow_initializer_t kernel_initializer_,
                              flexflow_initializer_t bias_initializer_,
                              char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor input = FFCObjectWrapper::unwrap_const(input_);
  Layer *shared_op = FFCObjectWrapper::unwrap(shared_op_);
  Initializer *kernel_initializer =
      FFCObjectWrapper::unwrap(kernel_initializer_);
  Initializer *bias_initializer = FFCObjectWrapper::unwrap(bias_initializer_);
  Tensor tensor = handle->conv2d(input,
                                 out_channels,
                                 kernel_h,
                                 kernel_w,
                                 stride_h,
                                 stride_w,
                                 padding_h,
                                 padding_w,
                                 activation,
                                 groups,
                                 use_bias,
                                 shared_op,
                                 kernel_initializer,
                                 bias_initializer,
                                 name);
  DEBUG_PRINT(
      "[Conv2d] new Tensor 4D %p (%d, %d, %d, %d), input %p, out_channels %d, "
      "kernel(%d, %d), stride(%d, %d), padding(%d, %d), activation %d, "
      "use_bias %d, shared_op %p, kernel_init %p, bias_init %p, name %s",
      tensor,
      tensor->dims[0],
      tensor->dims[1],
      tensor->dims[2],
      tensor->dims[3],
      input,
      out_channels,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      padding_h,
      padding_w,
      activation,
      use_bias,
      shared_op,
      kernel_initializer,
      bias_initializer,
      name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t
    flexflow_model_add_embedding(flexflow_model_t handle_,
                                 const flexflow_tensor_t input_,
                                 int num_entries,
                                 int out_dim,
                                 enum AggrMode aggr,
                                 DataType dtype,
                                 flexflow_op_t shared_op_,
                                 flexflow_initializer_t kernel_initializer_,
                                 char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor input = FFCObjectWrapper::unwrap_const(input_);
  Layer *shared_op = FFCObjectWrapper::unwrap(shared_op_);
  Initializer *kernel_initializer =
      FFCObjectWrapper::unwrap(kernel_initializer_);
  // TODO: update the flexflow_c and Python API to support other data types
  // Currently we assume it's float
  Tensor tensor = handle->embedding(input,
                                    num_entries,
                                    out_dim,
                                    aggr,
                                    dtype,
                                    shared_op,
                                    kernel_initializer,
                                    name);
  DEBUG_PRINT("[Embedding] new Tensor %p, input %p, num_entries %d, out_dim "
              "%d, aggr %d, dtype %d, shared_op %p, kernel_init %p, name %s",
              tensor,
              input,
              num_entries,
              out_dim,
              aggr,
              dtype,
              shared_op,
              kernel_initializer,
              name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t
    flexflow_model_add_pool2d(flexflow_model_t handle_,
                              flexflow_tensor_t input_,
                              int kernel_h,
                              int kernel_w,
                              int stride_h,
                              int stride_w,
                              int padding_h,
                              int padding_w,
                              enum PoolType type /* POOL_MAX */,
                              enum ActiMode activation /* AC_MODE_NONE */,
                              char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor input = FFCObjectWrapper::unwrap(input_);
  Tensor tensor = handle->pool2d(input,
                                 kernel_h,
                                 kernel_w,
                                 stride_h,
                                 stride_w,
                                 padding_h,
                                 padding_w,
                                 type,
                                 activation,
                                 name);
  DEBUG_PRINT(
      "[Pool2d] new Tensor 4D %p (%d, %d, %d, %d), input %p, kernel(%d, %d), "
      "stride(%d, %d), padding(%d, %d), pool %d, activation %d, name %s",
      tensor,
      tensor->dims[0],
      tensor->dims[1],
      tensor->dims[2],
      tensor->dims[3],
      input,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      padding_h,
      padding_w,
      type,
      activation,
      name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_batch_norm(flexflow_model_t handle_,
                                                const flexflow_tensor_t input_,
                                                bool relu,
                                                char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor input = FFCObjectWrapper::unwrap(input_);
  Tensor tensor = handle->batch_norm(input, relu, name);
  DEBUG_PRINT("[BatchNorm] new Tensor 4D %p (%d, %d, %d, %d), input %p, relu "
              "%d, name %s",
              tensor,
              tensor->dims[0],
              tensor->dims[1],
              tensor->dims[2],
              tensor->dims[3],
              input,
              relu,
              name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_layer_norm(flexflow_model_t handle_,
                                                const flexflow_tensor_t input_,
                                                int n,
                                                int *axes,
                                                bool elementwise_affine,
                                                float eps,
                                                bool use_bias,
                                                char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor input = FFCObjectWrapper::unwrap(input_);
  std::vector<int> axes_vec;
  for (int i = 0; i < n; i++) {
    axes_vec.push_back(axes[i]);
  }
  Tensor tensor = handle->layer_norm(input,
                                     axes_vec,
                                     elementwise_affine,
                                     eps,
                                     use_bias,
                                     input->data_type,
                                     name);
  DEBUG_PRINT("[LayerNorm] new Tensor %p, input %p, elementwise_affine %d, eps "
              "%f, name %s",
              tensor,
              input,
              elementwise_affine,
              eps,
              name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t *
    flexflow_model_add_residual_layer_norm(flexflow_model_t handle_,
                                           const flexflow_tensor_t input_,
                                           const flexflow_tensor_t residual1_,
                                           const flexflow_tensor_t residual2_,
                                           bool use_two_residuals,
                                           int n,
                                           int *axes,
                                           bool elementwise_affine,
                                           float eps,
                                           bool use_bias,
                                           bool inplace_residual,
                                           char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor input = FFCObjectWrapper::unwrap(input_);
  const Tensor residual1 = FFCObjectWrapper::unwrap(residual1_);
  const Tensor residual2 =
      use_two_residuals ? FFCObjectWrapper::unwrap(residual2_) : nullptr;
  Tensor tensor_outputs[2];
  std::vector<int> axes_vec;
  for (int i = 0; i < n; i++) {
    axes_vec.push_back(axes[i]);
  }
  if (use_two_residuals) {
    assert(residual2 != nullptr);
  }
  handle->residual_layer_norm(input,
                              residual1,
                              residual2,
                              tensor_outputs,
                              use_two_residuals,
                              axes_vec,
                              elementwise_affine,
                              eps,
                              use_bias,
                              inplace_residual,
                              input->data_type,
                              name);
  assert(tensor_outputs[0] != nullptr);
  assert(tensor_outputs[1] != nullptr);
  DEBUG_PRINT("[ResidualLayerNorm] input %p, residual1 %p, residual2 "
              "%p, output0: %p, "
              "output1: %p, use_two_residuals: %d, elementwise_affine %d, eps "
              "%f, use_bias: %d, inplace_residual: %d, name %s",
              input,
              residual1,
              residual2,
              tensor_outputs[0],
              tensor_outputs[1],
              use_two_residuals,
              elementwise_affine,
              eps,
              use_bias,
              inplace_residual,
              name);
  flexflow_tensor_t *tensor_outputs_wrapped =
      (flexflow_tensor_t *)calloc(2, sizeof(flexflow_tensor_t));
  tensor_outputs_wrapped[0] = FFCObjectWrapper::wrap(tensor_outputs[0]);
  tensor_outputs_wrapped[1] = FFCObjectWrapper::wrap(tensor_outputs[1]);
  return tensor_outputs_wrapped;
}

flexflow_tensor_t *flexflow_model_add_add_bias_residual_layer_norm(
    flexflow_model_t handle_,
    const flexflow_tensor_t input_,
    const flexflow_tensor_t residual_,
    int n,
    int *axes,
    bool elementwise_affine,
    float eps,
    bool use_bias,
    bool inplace_residual,
    char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor input = FFCObjectWrapper::unwrap(input_);
  const Tensor residual = FFCObjectWrapper::unwrap(residual_);
  Tensor tensor_outputs[2];
  std::vector<int> axes_vec;
  for (int i = 0; i < n; i++) {
    axes_vec.push_back(axes[i]);
  }
  handle->add_bias_residual_layer_norm(input,
                                       residual,
                                       tensor_outputs,
                                       axes_vec,
                                       elementwise_affine,
                                       eps,
                                       use_bias,
                                       inplace_residual,
                                       input->data_type,
                                       name);
  assert(tensor_outputs[0] != nullptr);
  assert(tensor_outputs[1] != nullptr);
  DEBUG_PRINT("[AddBiasResidualLayerNorm] input %p, residual %p, output0: %p, "
              "output1: %p, elementwise_affine %d, eps "
              "%f, use_bias %d, inplace_residual: %d, name %s",
              input,
              residual,
              tensor_outputs[0],
              tensor_outputs[1],
              elementwise_affine,
              eps,
              use_bias,
              inplace_residual,
              name);
  flexflow_tensor_t *tensor_outputs_wrapped =
      (flexflow_tensor_t *)calloc(2, sizeof(flexflow_tensor_t));
  tensor_outputs_wrapped[0] = FFCObjectWrapper::wrap(tensor_outputs[0]);
  tensor_outputs_wrapped[1] = FFCObjectWrapper::wrap(tensor_outputs[1]);
  return tensor_outputs_wrapped;
}

flexflow_tensor_t
    flexflow_model_add_sigmoid_silu_multi(flexflow_model_t handle_,
                                          const flexflow_tensor_t input1_,
                                          const flexflow_tensor_t input2_,
                                          char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor input1 = FFCObjectWrapper::unwrap(input1_);
  const Tensor input2 = FFCObjectWrapper::unwrap(input2_);
  Tensor tensor =
      handle->sigmoid_silu_multi(input1, input2, input1->data_type, name);
  DEBUG_PRINT("[SigmoidSiluMulti] new Tensor %p, input1 %p, input2 %p, name %s",
              tensor,
              input1,
              input2,
              name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_batch_matmul(flexflow_model_t handle_,
                                                  const flexflow_tensor_t a_,
                                                  const flexflow_tensor_t b_,
                                                  int a_seq_length_dim,
                                                  int b_seq_length_dim) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor a = FFCObjectWrapper::unwrap(a_);
  Tensor b = FFCObjectWrapper::unwrap(b_);
  Tensor tensor =
      handle->batch_matmul(a, b, a_seq_length_dim, b_seq_length_dim);
  DEBUG_PRINT("[BatchMatMul] new Tensor %p, a %p, b %p", tensor, a, b);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_dense(
    flexflow_model_t handle_,
    const flexflow_tensor_t input_,
    int out_dim,
    enum ActiMode activation /* AC_MODE_NONE */,
    bool use_bias /* true */,
    enum DataType data_type /*DT_FLOAT*/,
    flexflow_op_t shared_op_,
    flexflow_initializer_t kernel_initializer_,
    flexflow_initializer_t bias_initializer_,
    enum RegularizerMode kernel_reg_type /* REG_MODE_NONE */,
    float kernel_reg_lambda,
    char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  const Tensor input = FFCObjectWrapper::unwrap_const(input_);
  Layer *shared_op = FFCObjectWrapper::unwrap(shared_op_);
  Initializer *kernel_initializer =
      FFCObjectWrapper::unwrap(kernel_initializer_);
  Initializer *bias_initializer = FFCObjectWrapper::unwrap(bias_initializer_);
  Tensor tensor = handle->dense(input,
                                out_dim,
                                activation,
                                use_bias,
                                data_type,
                                shared_op,
                                kernel_initializer,
                                bias_initializer,
                                kernel_reg_type,
                                kernel_reg_lambda,
                                name);
  DEBUG_PRINT("[Dense] new Tensor 2D %p (%d, %d, %d, %d), input %p, out_dim "
              "%d, activation %d, use_bias %d, shared_op %p, kernel_init %p, "
              "bias_init %p, name %s",
              tensor,
              tensor->dims[0],
              tensor->dims[1],
              tensor->dims[2],
              tensor->dims[3],
              input,
              out_dim,
              activation,
              use_bias,
              shared_op,
              kernel_initializer,
              bias_initializer,
              name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_concat(flexflow_model_t handle_,
                                            int n,
                                            flexflow_tensor_t *input_,
                                            int axis,
                                            char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor tensor;
  std::vector<Tensor> input_vec;
  char cbuffer[256];
  char *cbuffer_ptr = cbuffer;
  sprintf(cbuffer_ptr, "[Concat] input tensor");
  cbuffer_ptr += 21;
  for (int i = 0; i < n; i++) {
    Tensor t = FFCObjectWrapper::unwrap(input_[i]);
    input_vec.push_back(t);
    if (i < 10) {
      sprintf(cbuffer_ptr, "%p ", t);
      cbuffer_ptr += 15;
    }
  }
  tensor = handle->concat(n, input_vec.data(), axis, name);
  sprintf(cbuffer_ptr, ", concat new Tensor %p", tensor);
  DEBUG_PRINT("%s, n %d, axis %d, name %s", cbuffer, n, axis, name);
  return FFCObjectWrapper::wrap(tensor);
}

void flexflow_model_add_split(flexflow_model_t handle_,
                              flexflow_tensor_t input_,
                              int n,
                              flexflow_tensor_t *outputs_,
                              int *split,
                              int axis,
                              char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor input = FFCObjectWrapper::unwrap(input_);
  std::vector<int> split_vec;
  Tensor *outputs = new Tensor[n];
  for (int i = 0; i < n; i++) {
    split_vec.push_back(split[i]);
  }
  handle->split(input, outputs, split_vec, axis, name);
  for (int i = 0; i < n; i++) {
    outputs_[i] = FFCObjectWrapper::wrap(outputs[i]);
  }
  char cbuffer[256];
  char *cbuffer_ptr = cbuffer;
  sprintf(cbuffer_ptr, "[Split] input tensor %p output tensors ", input);
  cbuffer_ptr += 51;
  for (int i = 0; i < n; i++) {
    sprintf(cbuffer_ptr, "%p ", outputs_[i].impl);
    cbuffer_ptr += 15;
    if (i >= 10) {
      break;
    }
  }
  DEBUG_PRINT("%s, n %d, axis %d, name %s", cbuffer, n, axis, name);
  delete[] outputs;
}

flexflow_tensor_t flexflow_model_add_flat(flexflow_model_t handle_,
                                          flexflow_tensor_t input_,
                                          char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor input = FFCObjectWrapper::unwrap(input_);
  Tensor tensor = handle->flat(input, name);
  DEBUG_PRINT(
      "[Flat] new Tensor 4D %p, input %p, name %s", tensor, input, name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_gather(flexflow_model_t handle_,
                                            const flexflow_tensor_t input_,
                                            const flexflow_tensor_t index_,
                                            int dim,
                                            char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor input = FFCObjectWrapper::unwrap(input_);
  Tensor index = FFCObjectWrapper::unwrap(index_);
  Tensor tensor = handle->gather(input, index, dim, name);
  DEBUG_PRINT("[Gather] new Tensor %p, input %p, index %p, dim %d name %s",
              tensor,
              input,
              index,
              dim,
              name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_softmax(flexflow_model_t handle_,
                                             const flexflow_tensor_t input_,
                                             int dim,
                                             char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor input = FFCObjectWrapper::unwrap(input_);
  Tensor tensor = handle->softmax(input, dim, input->data_type, name);
  DEBUG_PRINT(
      "[Softmax] new Tensor %p, input %p, name %s", tensor, input, name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_transpose(flexflow_model_t handle_,
                                               const flexflow_tensor_t input_,
                                               int n,
                                               int *perm,
                                               char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor input = FFCObjectWrapper::unwrap(input_);
  std::vector<int> perm_vec;
  for (int i = 0; i < n; i++) {
    perm_vec.push_back(perm[i]);
  }
  Tensor tensor = handle->transpose(input, perm_vec, name);
  DEBUG_PRINT("[Transpose] new Tensor %p, input %p, n %d, name %s",
              tensor,
              input,
              n,
              name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_reshape(flexflow_model_t handle_,
                                             const flexflow_tensor_t input_,
                                             int n,
                                             int *shape,
                                             char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor input = FFCObjectWrapper::unwrap(input_);
  std::vector<int> shape_vec;
  for (int i = 0; i < n; i++) {
    shape_vec.push_back(shape[i]);
  }
  Tensor tensor = handle->reshape(input, shape_vec, name);
  DEBUG_PRINT("[Reshape] new Tensor %p, input %p, n %d, name %s",
              tensor,
              input,
              n,
              name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_reverse(flexflow_model_t handle_,
                                             const flexflow_tensor_t input_,
                                             int axis,
                                             char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor input = FFCObjectWrapper::unwrap(input_);
  Tensor tensor = handle->reverse(input, axis, name);
  DEBUG_PRINT("[Reverse] new Tensor %p, input %p, axis %d, name %s",
              tensor,
              input,
              axis,
              name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t
    flexflow_model_add_scalar_multiply(flexflow_model_t handle_,
                                       const flexflow_tensor_t input_,
                                       float const scalar,
                                       bool inplace,
                                       char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor input = FFCObjectWrapper::unwrap(input_);
  Tensor tensor = handle->scalar_multiply(input, scalar, inplace, name);
  DEBUG_PRINT("[Scalar multiply] new Tensor %p, input %p, scalar %f, name %s",
              tensor,
              input,
              scalar,
              name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_scalar_add(flexflow_model_t handle_,
                                                const flexflow_tensor_t input_,
                                                float const scalar,
                                                bool inplace,
                                                char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor input = FFCObjectWrapper::unwrap(input_);
  Tensor tensor = handle->scalar_add(input, scalar, inplace, name);
  DEBUG_PRINT("[Scalar addition] new Tensor %p, input %p, scalar %f, name %s",
              tensor,
              input,
              scalar,
              name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_scalar_sub(flexflow_model_t handle_,
                                                const flexflow_tensor_t input_,
                                                float const scalar,
                                                bool inplace,
                                                char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor input = FFCObjectWrapper::unwrap(input_);
  Tensor tensor = handle->scalar_sub(input, scalar, inplace, name);
  DEBUG_PRINT(
      "[Scalar subtraction] new Tensor %p, input %p, scalar %f, name %s",
      tensor,
      input,
      scalar,
      name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t
    flexflow_model_add_scalar_truediv(flexflow_model_t handle_,
                                      const flexflow_tensor_t input_,
                                      float const scalar,
                                      bool inplace,
                                      char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor input = FFCObjectWrapper::unwrap(input_);
  Tensor tensor = handle->scalar_truediv(input, scalar, inplace, name);
  DEBUG_PRINT(
      "[Scalar true division] new Tensor %p, input %p, scalar %f, name %s",
      tensor,
      input,
      scalar,
      name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_gelu(flexflow_model_t handle_,
                                          const flexflow_tensor_t input_,
                                          char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor input = FFCObjectWrapper::unwrap(input_);
  Tensor tensor = handle->gelu(input, name);
  DEBUG_PRINT("[GeLU] new Tensor %p, input %p, name %s", tensor, input, name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_identity(flexflow_model_t handle_,
                                              const flexflow_tensor_t input_,
                                              char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor input = FFCObjectWrapper::unwrap(input_);
  Tensor tensor = handle->identity(input, name);
  DEBUG_PRINT(
      "[Identity] new Tensor %p, input %p, name %s", tensor, input, name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_relu(flexflow_model_t handle_,
                                          const flexflow_tensor_t input_,
                                          bool inplace,
                                          char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor input = FFCObjectWrapper::unwrap(input_);
  Tensor tensor = handle->relu(input, name);
  DEBUG_PRINT("[Relu] new Tensor %p, input %p, name %s", tensor, input, name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_sigmoid(flexflow_model_t handle_,
                                             const flexflow_tensor_t input_,
                                             char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor input = FFCObjectWrapper::unwrap(input_);
  Tensor tensor = handle->sigmoid(input, name);
  DEBUG_PRINT(
      "[Sigmoid] new Tensor %p, input %p, name %s", tensor, input, name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_tanh(flexflow_model_t handle_,
                                          const flexflow_tensor_t input_,
                                          char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor input = FFCObjectWrapper::unwrap(input_);
  Tensor tensor = handle->tanh(input, name);
  DEBUG_PRINT("[Tanh] new Tensor %p, input %p, name %s", tensor, input, name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_elu(flexflow_model_t handle_,
                                         const flexflow_tensor_t input_,
                                         bool inplace,
                                         char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor input = FFCObjectWrapper::unwrap(input_);
  Tensor tensor = handle->elu(input, name);
  DEBUG_PRINT("[Elu] new Tensor %p, input %p, name %s", tensor, input, name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_dropout(flexflow_model_t handle_,
                                             const flexflow_tensor_t input_,
                                             float rate,
                                             unsigned long long seed,
                                             char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor input = FFCObjectWrapper::unwrap(input_);
  Tensor tensor = handle->dropout(input, rate, seed, name);
  DEBUG_PRINT("[Dropout] new Tensor %p, input %p, rate %f, seed %lld, name %s",
              tensor,
              input,
              rate,
              seed,
              name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_multihead_attention(
    flexflow_model_t handle_,
    const flexflow_tensor_t query_,
    const flexflow_tensor_t key_,
    const flexflow_tensor_t value_,
    int embed_dim,
    int num_heads,
    int kdim,
    int vdim,
    float dropout,
    bool bias,
    bool add_bias_kv,
    bool add_zero_attn,
    flexflow_initializer_t kernel_initializer_,
    char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor query = FFCObjectWrapper::unwrap(query_);
  Tensor key = FFCObjectWrapper::unwrap(key_);
  Tensor value = FFCObjectWrapper::unwrap(value_);
  Initializer *kernel_initializer =
      FFCObjectWrapper::unwrap(kernel_initializer_);
  Tensor tensor = handle->multihead_attention(query,
                                              key,
                                              value,
                                              embed_dim,
                                              num_heads,
                                              kdim,
                                              vdim,
                                              dropout,
                                              bias,
                                              add_bias_kv,
                                              add_zero_attn,
                                              query->data_type,
                                              kernel_initializer,
                                              name);
  DEBUG_PRINT("[MultiHeadAttention] new Tensor %p, query %p, key %p, value %p, "
              "embed_dim %d, num_heads %d, kdim %d, vdim %d, dropout %f, bias "
              "%d, add_bias_kv %d, add_zero_attn %d, kernel_init %p, name %s",
              tensor,
              query,
              key,
              value,
              embed_dim,
              num_heads,
              kdim,
              vdim,
              dropout,
              bias,
              add_bias_kv,
              add_zero_attn,
              kernel_initializer,
              name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_inc_multihead_self_attention(
    flexflow_model_t handle_,
    const flexflow_tensor_t input_,
    int embed_dim,
    int num_heads,
    int kdim,
    int vdim,
    float dropout,
    bool bias,
    bool add_bias_kv,
    bool add_zero_attn,
    enum DataType data_type,
    flexflow_initializer_t kernel_initializer_,
    bool apply_rotary_embedding,
    bool scaling_query,
    float scaling_factor,
    bool qk_prod_scaling,
    bool position_bias,
    char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor input = FFCObjectWrapper::unwrap(input_);
  Initializer *kernel_initializer =
      FFCObjectWrapper::unwrap(kernel_initializer_);
  Tensor tensor = handle->inc_multihead_self_attention(input,
                                                       embed_dim,
                                                       num_heads,
                                                       kdim,
                                                       vdim,
                                                       dropout,
                                                       bias,
                                                       add_bias_kv,
                                                       add_zero_attn,
                                                       data_type,
                                                       kernel_initializer,
                                                       apply_rotary_embedding,
                                                       scaling_query,
                                                       scaling_factor,
                                                       qk_prod_scaling,
                                                       position_bias,
                                                       name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_spec_inc_multihead_self_attention(
    flexflow_model_t handle_,
    const flexflow_tensor_t input_,
    int embed_dim,
    int num_heads,
    int kdim,
    int vdim,
    float dropout,
    bool bias,
    bool add_bias_kv,
    bool add_zero_attn,
    enum DataType data_type,
    flexflow_initializer_t kernel_initializer_,
    bool apply_rotary_embedding,
    bool scaling_query,
    float scaling_factor,
    bool qk_prod_scaling,
    bool position_bias,
    char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor input = FFCObjectWrapper::unwrap(input_);
  Initializer *kernel_initializer =
      FFCObjectWrapper::unwrap(kernel_initializer_);
  Tensor tensor =
      handle->spec_inc_multihead_self_attention(input,
                                                embed_dim,
                                                num_heads,
                                                kdim,
                                                vdim,
                                                dropout,
                                                bias,
                                                add_bias_kv,
                                                add_zero_attn,
                                                data_type,
                                                kernel_initializer,
                                                apply_rotary_embedding,
                                                scaling_query,
                                                scaling_factor,
                                                qk_prod_scaling,
                                                position_bias,
                                                name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_inc_multihead_self_attention_verify(
    flexflow_model_t handle_,
    const flexflow_tensor_t input_,
    int embed_dim,
    int num_heads,
    int kdim,
    int vdim,
    float dropout,
    bool bias,
    bool add_bias_kv,
    bool add_zero_attn,
    enum DataType data_type,
    flexflow_initializer_t kernel_initializer_,
    bool apply_rotary_embedding,
    bool scaling_query,
    float scaling_factor,
    bool qk_prod_scaling,
    bool position_bias,
    char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor input = FFCObjectWrapper::unwrap(input_);
  Initializer *kernel_initializer =
      FFCObjectWrapper::unwrap(kernel_initializer_);
  Tensor tensor =
      handle->inc_multihead_self_attention_verify(input,
                                                  embed_dim,
                                                  num_heads,
                                                  kdim,
                                                  vdim,
                                                  dropout,
                                                  bias,
                                                  add_bias_kv,
                                                  add_zero_attn,
                                                  data_type,
                                                  kernel_initializer,
                                                  apply_rotary_embedding,
                                                  scaling_query,
                                                  scaling_factor,
                                                  qk_prod_scaling,
                                                  position_bias,
                                                  name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_inc_multiquery_self_attention(
    flexflow_model_t handle_,
    const flexflow_tensor_t input_,
    int embed_dim,
    int num_q_heads,
    int num_kv_heads,
    int kdim,
    int vdim,
    float dropout,
    bool bias,
    bool add_bias_kv,
    bool add_zero_attn,
    enum DataType data_type,
    flexflow_initializer_t kernel_initializer_,
    bool apply_rotary_embedding,
    bool scaling_query,
    float scaling_factor,
    bool qk_prod_scaling,
    bool position_bias,
    char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor input = FFCObjectWrapper::unwrap(input_);
  Initializer *kernel_initializer =
      FFCObjectWrapper::unwrap(kernel_initializer_);
  Tensor tensor = handle->inc_multiquery_self_attention(input,
                                                        embed_dim,
                                                        num_q_heads,
                                                        num_kv_heads,
                                                        kdim,
                                                        vdim,
                                                        dropout,
                                                        bias,
                                                        add_bias_kv,
                                                        add_zero_attn,
                                                        data_type,
                                                        kernel_initializer,
                                                        apply_rotary_embedding,
                                                        scaling_query,
                                                        scaling_factor,
                                                        qk_prod_scaling,
                                                        position_bias,
                                                        name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_spec_inc_multiquery_self_attention(
    flexflow_model_t handle_,
    const flexflow_tensor_t input_,
    int embed_dim,
    int num_q_heads,
    int num_kv_heads,
    int kdim,
    int vdim,
    float dropout,
    bool bias,
    bool add_bias_kv,
    bool add_zero_attn,
    enum DataType data_type,
    flexflow_initializer_t kernel_initializer_,
    bool apply_rotary_embedding,
    bool scaling_query,
    float scaling_factor,
    bool qk_prod_scaling,
    bool position_bias,
    char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor input = FFCObjectWrapper::unwrap(input_);
  Initializer *kernel_initializer =
      FFCObjectWrapper::unwrap(kernel_initializer_);
  Tensor tensor =
      handle->spec_inc_multiquery_self_attention(input,
                                                 embed_dim,
                                                 num_q_heads,
                                                 num_kv_heads,
                                                 kdim,
                                                 vdim,
                                                 dropout,
                                                 bias,
                                                 add_bias_kv,
                                                 add_zero_attn,
                                                 data_type,
                                                 kernel_initializer,
                                                 apply_rotary_embedding,
                                                 scaling_query,
                                                 scaling_factor,
                                                 qk_prod_scaling,
                                                 position_bias,
                                                 name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_inc_multiquery_self_attention_verify(
    flexflow_model_t handle_,
    const flexflow_tensor_t input_,
    int embed_dim,
    int num_q_heads,
    int num_kv_heads,
    int kdim,
    int vdim,
    float dropout,
    bool bias,
    bool add_bias_kv,
    bool add_zero_attn,
    enum DataType data_type,
    flexflow_initializer_t kernel_initializer_,
    bool apply_rotary_embedding,
    bool scaling_query,
    float scaling_factor,
    bool qk_prod_scaling,
    bool position_bias,
    char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor input = FFCObjectWrapper::unwrap(input_);
  Initializer *kernel_initializer =
      FFCObjectWrapper::unwrap(kernel_initializer_);
  Tensor tensor =
      handle->inc_multiquery_self_attention_verify(input,
                                                   embed_dim,
                                                   num_q_heads,
                                                   num_kv_heads,
                                                   kdim,
                                                   vdim,
                                                   dropout,
                                                   bias,
                                                   add_bias_kv,
                                                   add_zero_attn,
                                                   data_type,
                                                   kernel_initializer,
                                                   apply_rotary_embedding,
                                                   scaling_query,
                                                   scaling_factor,
                                                   qk_prod_scaling,
                                                   position_bias,
                                                   name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_rms_norm(flexflow_model_t handle_,
                                              const flexflow_tensor_t input_,
                                              float eps,
                                              int dim,
                                              char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor input = FFCObjectWrapper::unwrap(input_);
  Tensor tensor = handle->rms_norm(input, eps, dim, input->data_type, name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t *
    flexflow_model_add_residual_rms_norm(flexflow_model_t handle_,
                                         const flexflow_tensor_t input1_,
                                         const flexflow_tensor_t input2_,
                                         float eps,
                                         int dim,
                                         bool inplace_residual,
                                         char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor input1 = FFCObjectWrapper::unwrap(input1_);
  Tensor input2 = FFCObjectWrapper::unwrap(input2_);
  Tensor tensor_outputs[2];
  handle->residual_rms_norm(input1,
                            input2,
                            tensor_outputs,
                            eps,
                            dim,
                            inplace_residual,
                            input1->data_type,
                            name);
  assert(tensor_outputs[0] != nullptr);
  assert(tensor_outputs[1] != nullptr);
  flexflow_tensor_t *tensor_outputs_wrapped =
      (flexflow_tensor_t *)calloc(2, sizeof(flexflow_tensor_t));
  tensor_outputs_wrapped[0] = FFCObjectWrapper::wrap(tensor_outputs[0]);
  tensor_outputs_wrapped[1] = FFCObjectWrapper::wrap(tensor_outputs[1]);
  return tensor_outputs_wrapped;
}

flexflow_tensor_t flexflow_model_add_arg_top_k(flexflow_model_t handle_,
                                               const flexflow_tensor_t input_,
                                               int k,
                                               bool sorted,
                                               bool speculative_decoding,
                                               char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor input = FFCObjectWrapper::unwrap(input_);
  Tensor tensor =
      handle->arg_top_k(input, k, sorted, speculative_decoding, name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_beam_top_k(flexflow_model_t handle_,
                                                const flexflow_tensor_t input_,
                                                int max_beam_size,
                                                bool sorted,
                                                char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor input = FFCObjectWrapper::unwrap(input_);
  Tensor tensor = handle->beam_top_k(input, max_beam_size, sorted, name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_sampling(flexflow_model_t handle_,
                                              const flexflow_tensor_t input_,
                                              float top_p,
                                              char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor input = FFCObjectWrapper::unwrap(input_);
  Tensor tensor = handle->sampling(input, top_p, name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_tensor_t flexflow_model_add_argmax(flexflow_model_t handle_,
                                            const flexflow_tensor_t input_,
                                            bool beam_search,
                                            char const *name) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor input = FFCObjectWrapper::unwrap(input_);
  Tensor tensor = handle->argmax(input, beam_search, name);
  return FFCObjectWrapper::wrap(tensor);
}

flexflow_peft_model_id_t flexflow_model_add_lora_layer(
    flexflow_model_t handle_,
    const flexflow_lora_linear_config_t peft_config_) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  LoraLinearConfig const *peft_config = FFCObjectWrapper::unwrap(peft_config_);
  PEFTModelID *peft_model_id = handle->add_lora_layer(*peft_config);

  DEBUG_PRINT("[Add Lora Layer] model handle: %p, peft_config handle %p, "
              "peft_model_id: %p",
              handle,
              peft_config,
              peft_model_id);
  return FFCObjectWrapper::wrap(peft_model_id);
}

void flexflow_model_set_sgd_optimizer(flexflow_model_t handle_,
                                      flexflow_sgd_optimizer_t optimizer_) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  SGDOptimizer *optimizer = FFCObjectWrapper::unwrap(optimizer_);
  handle->optimizer = static_cast<Optimizer *>(optimizer);
}

void flexflow_model_set_adam_optimizer(flexflow_model_t handle_,
                                       flexflow_adam_optimizer_t optimizer_) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  AdamOptimizer *optimizer = FFCObjectWrapper::unwrap(optimizer_);
  handle->optimizer = static_cast<Optimizer *>(optimizer);
}

void flexflow_model_print_layers(flexflow_model_t handle_, int id) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  handle->print_layers(id);
}

flexflow_op_t flexflow_model_get_layer_by_id(flexflow_model_t handle_,
                                             int layer_id) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Layer *layer = handle->layers[layer_id];
  return FFCObjectWrapper::wrap(layer);
}

flexflow_op_t flexflow_model_get_last_layer(flexflow_model_t handle_) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  Layer *layer = handle->layers.back();
  return FFCObjectWrapper::wrap(layer);
}

flexflow_tensor_t flexflow_model_get_parameter_by_id(flexflow_model_t handle_,
                                                     int layer_id) {
  assert(false);
}

flexflow_perf_metrics_t
    flexflow_model_get_perf_metrics(flexflow_model_t handle_) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  PerfMetrics *perf_metrics = new PerfMetrics();
  *perf_metrics = handle->current_metrics.get_result<PerfMetrics>();
  DEBUG_PRINT("[Model] create PerfMetrics %p, train_correct %d",
              perf_metrics,
              perf_metrics->train_correct);
  return FFCObjectWrapper::wrap(perf_metrics);
}

void flexflow_model_set_transformer_layer_id(flexflow_model_t handle_, int id) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  handle->set_transformer_layer_id(id);
}

void flexflow_model_generate(flexflow_model_t handle_,
                             int num_requests,
                             enum RequestType *request_types,
                             char const **input_texts,
                             char **output_texts,
                             int *max_seq_lengths,
                             flexflow_peft_model_id_t *peft_model_ids,
                             char const **dataset_filepaths,
                             int *training_steps;
                             int **output_length_and_tokens) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  std::vector<Request> requests;

  int finetuning_req_idx = 0; 
  for (int i = 0; i < num_requests; i++) {
    if (request_types[i] == RequestType::REQ_INFERENCE) {
      std::string const text_str(input_texts[i]);
      Request inference_req;
      inference_req.prompt = text_str;
      inference_req.max_sequence_length = max_seq_lengths[i];
      if (peft_model_ids[i] != nullptr) {
        PEFTModelID *peft_model_id = FFCObjectWrapper::unwrap(peft_model_ids[i]);
        inference_req.peft_model_id = *peft_model_id;
      }
      requests.push_back(inference_req);
      DEBUG_PRINT("[Model] generate[%d] %p %s %i",
                  i,
                  handle,
                  text_str.c_str(),
                  max_seq_lengths[i]);
    } else {
      Request fine_tuning_req;
      fine_tuning_req.req_type = RequestType::REQ_FINETUNING;
      fine_tuning_req.max_sequence_length = max_seq_lengths[i];
      if (peft_model_ids[i] != nullptr) {
        PEFTModelID *peft_model_id = FFCObjectWrapper::unwrap(peft_model_ids[i]);
        fine_tuning_req.peft_model_id = *peft_model_id;
      }
      std::string const dataset_fp(dataset_filepaths[finetuning_req_idx]);
      fine_tuning_req.dataset_filepath = dataset_fp;
      fine_tuning_req.max_training_steps = training_steps[finetuning_req_idx];
      requests.push_back(finetuning_req_idx);
      DEBUG_PRINT("[Model] generate[%d] %p %s %i %i",
                  i,
                  handle,
                  dataset_fp.c_str(),
                  max_seq_lengths[i],
                  training_steps[finetuning_req_idx]);
      finetuning_req_idx++;
    }
  }

  std::vector<GenerationResult> results = handle->generate(requests);

  for (int i = 0; i < num_requests; i++) {
    if (request_types[i] == RequestType::REQ_INFERENCE) {
      // If the prompt exceeds max seq len, check that we return the prompt with no
      // additional token. Otherwise, check that the output does not exceed the max
      // sequence length.
      assert(results[i].output_tokens.size() <= max_seq_length ||
            results[i].output_tokens.size() == results[i].input_tokens.size());
      output_length_and_tokens[i][0] = results[i].output_tokens.size();
      std::copy(results[i].output_tokens.begin(),
                results[i].output_tokens.end(),
                output_length_and_tokens[i] + 1);
      std::memcpy(output_texts[i],
                  results[i].output_text.c_str(),
                  results[i].output_text.length());
    }
  }
}

void flexflow_model_set_position_offset(flexflow_model_t handle_,
                                        int const offset) {
  FFModel *handle = FFCObjectWrapper::unwrap(handle_);
  handle->set_position_offset(offset);
}

// -----------------------------------------------------------------------
// Tensor
// -----------------------------------------------------------------------

flexflow_tensor_t flexflow_tensor_create(flexflow_model_t model_,
                                         int num_dims,
                                         int const *dims,
                                         enum DataType data_type,
                                         bool create_grad /* true */) {
  Tensor tensor;
  FFModel *model = FFCObjectWrapper::unwrap(model_);
  if (num_dims == 1) {
    tensor = model->create_tensor<1>(dims, data_type, NULL, create_grad);
    DEBUG_PRINT("[Tensor] new 1D %p (%d, %d, %d, %d)",
                tensor,
                tensor->dims[0],
                tensor->dims[1],
                tensor->dims[2],
                tensor->dims[3]);
  } else if (num_dims == 2) {
    tensor = model->create_tensor<2>(dims, data_type, NULL, create_grad);
    DEBUG_PRINT("[Tensor] new 2D %p (%d, %d, %d, %d)",
                tensor,
                tensor->dims[0],
                tensor->dims[1],
                tensor->dims[2],
                tensor->dims[3]);
  } else if (num_dims == 3) {
    tensor = model->create_tensor<3>(dims, data_type, NULL, create_grad);
    DEBUG_PRINT("[Tensor] new 3D %p (%d, %d, %d, %d)",
                tensor,
                tensor->dims[0],
                tensor->dims[1],
                tensor->dims[2],
                tensor->dims[3]);
  } else if (num_dims == 4) {
    tensor = model->create_tensor<4>(dims, data_type, NULL, create_grad);
    DEBUG_PRINT("[Tensor] new 4D %p (%d, %d, %d, %d)",
                tensor,
                tensor->dims[0],
                tensor->dims[1],
                tensor->dims[2],
                tensor->dims[3]);
#if MAX_TENSOR_DIM >= 5
  } else if (num_dims == 5) {
    tensor = model->create_tensor<5>(dims, data_type, NULL, create_grad);
    DEBUG_PRINT("[Tensor] new 5D %p (%d, %d, %d, %d, %d)",
                tensor,
                tensor->dims[0],
                tensor->dims[1],
                tensor->dims[2],
                tensor->dims[3],
                tensor->dims[4]);
#endif
  } else {
    assert(0);
  }
  // printf("[create_tensor()] %d %d %d\n",
  // tensor->region.get_index_space().get_id(),
  // tensor->region.get_field_space().get_id(), tensor->region.get_tree_id());
  return FFCObjectWrapper::wrap(tensor);
}

void flexflow_tensor_map(flexflow_model_t model_,
                         flexflow_tensor_t tensor_,
                         flexflow_op_t op_) {
  assert(false);
}

flexflow_tensor_t flexflow_constant_create(flexflow_model_t model_,
                                           int num_dims,
                                           int const *dims,
                                           float value,
                                           enum DataType data_type) {
  Tensor tensor;
  FFModel *model = FFCObjectWrapper::unwrap(model_);
  if (num_dims == 1) {
    tensor = model->create_constant<1>(dims, value, data_type);
    DEBUG_PRINT("[Tensor] new 1D %p (%d, %d, %d, %d)",
                tensor,
                tensor->dims[0],
                tensor->dims[1],
                tensor->dims[2],
                tensor->dims[3]);
  } else if (num_dims == 2) {
    tensor = model->create_constant<2>(dims, value, data_type);
    DEBUG_PRINT("[Tensor] new 2D %p (%d, %d, %d, %d)",
                tensor,
                tensor->dims[0],
                tensor->dims[1],
                tensor->dims[2],
                tensor->dims[3]);
  } else if (num_dims == 3) {
    tensor = model->create_constant<3>(dims, value, data_type);
    DEBUG_PRINT("[Tensor] new 3D %p (%d, %d, %d, %d)",
                tensor,
                tensor->dims[0],
                tensor->dims[1],
                tensor->dims[2],
                tensor->dims[3]);
  } else if (num_dims == 4) {
    tensor = model->create_constant<4>(dims, value, data_type);
    DEBUG_PRINT("[Tensor] new 4D %p (%d, %d, %d, %d)",
                tensor,
                tensor->dims[0],
                tensor->dims[1],
                tensor->dims[2],
                tensor->dims[3]);
#if MAX_TENSOR_DIM >= 5
  } else if (num_dims == 5) {
    tensor = model->create_constant<5>(dims, value, data_type);
    DEBUG_PRINT("[Tensor] new 5D %p (%d, %d, %d, %d, %d)",
                tensor,
                tensor->dims[0],
                tensor->dims[1],
                tensor->dims[2],
                tensor->dims[3],
                tensor->dims[4]);
#endif
  } else {
    assert(0);
  }
  return FFCObjectWrapper::wrap(tensor);
}

void flexflow_tensor_destroy(flexflow_tensor_t handle_) {
  Tensor handle = FFCObjectWrapper::unwrap(handle_);
  DEBUG_PRINT("[Tensor] delete %p", handle);
  delete handle;
}

void flexflow_tensor_inline_map(flexflow_tensor_t handle_,
                                flexflow_model_t model_,
                                flexflow_config_t config_) {
  Tensor handle = FFCObjectWrapper::unwrap(handle_);
  FFModel *model = FFCObjectWrapper::unwrap(model_);
  ParallelTensor tensor;
  model->get_parallel_tensor_from_tensor(handle, tensor);
  FFConfig *config = FFCObjectWrapper::unwrap(config_);
  tensor->inline_map(*config);
}

void flexflow_tensor_inline_unmap(flexflow_tensor_t handle_,
                                  flexflow_model_t model_,
                                  flexflow_config_t config_) {
  Tensor handle = FFCObjectWrapper::unwrap(handle_);
  FFModel *model = FFCObjectWrapper::unwrap(model_);
  ParallelTensor tensor;
  model->get_parallel_tensor_from_tensor(handle, tensor);
  FFConfig *config = FFCObjectWrapper::unwrap(config_);
  tensor->inline_unmap(*config);
}

float *flexflow_tensor_get_raw_ptr_float(flexflow_tensor_t handle_,
                                         flexflow_model_t model_,
                                         flexflow_config_t config_) {
  Tensor handle = FFCObjectWrapper::unwrap(handle_);
  FFModel *model = FFCObjectWrapper::unwrap(model_);
  FFConfig *config = FFCObjectWrapper::unwrap(config_);
  ParallelTensor ptensor;
  model->get_parallel_tensor_from_tensor(handle, ptensor);
  float *raw_ptr = ptensor->get_raw_ptr<float>(*config);
  return raw_ptr;
}

int32_t *flexflow_tensor_get_raw_ptr_int32(flexflow_tensor_t handle_,
                                           flexflow_model_t model_,
                                           flexflow_config_t config_) {
  Tensor handle = FFCObjectWrapper::unwrap(handle_);
  FFModel *model = FFCObjectWrapper::unwrap(model_);
  FFConfig *config = FFCObjectWrapper::unwrap(config_);
  ParallelTensor ptensor;
  model->get_parallel_tensor_from_tensor(handle, ptensor);
  int32_t *raw_ptr = ptensor->get_raw_ptr<int32_t>(*config);
  return raw_ptr;
}

int flexflow_tensor_get_num_dims(flexflow_tensor_t handle_) {
  Tensor handle = FFCObjectWrapper::unwrap(handle_);
  return handle->num_dims;
}

int flexflow_tensor_get_dim(flexflow_tensor_t handle_, int legion_axis) {
  Tensor handle = FFCObjectWrapper::unwrap(handle_);
  DEBUG_PRINT("[Tensor] get dims [%d, %d, %d, %d]",
              handle->dims[3],
              handle->dims[2],
              handle->dims[1],
              handle->dims[0]);
  return handle->dims[legion_axis];
}

int *flexflow_tensor_get_dims(flexflow_tensor_t handle_) {
  Tensor handle = FFCObjectWrapper::unwrap(handle_);
  DEBUG_PRINT("[Tensor] get dims [%d, %d, %d, %d]",
              handle->dims[3],
              handle->dims[2],
              handle->dims[1],
              handle->dims[0]);
  return &(handle->dims[0]);
}

int flexflow_tensor_get_data_type(flexflow_tensor_t handle_) {
  Tensor handle = FFCObjectWrapper::unwrap(handle_);
  return static_cast<int>(handle->data_type);
}

flexflow_op_t flexflow_tensor_get_owner_op(flexflow_tensor_t handle_) {
  Tensor handle = FFCObjectWrapper::unwrap(handle_);
  return FFCObjectWrapper::wrap_const(handle->owner_layer);
}

void flexflow_tensor_attach_raw_ptr(flexflow_tensor_t handle_,
                                    flexflow_model_t model_,
                                    flexflow_config_t config_,
                                    void *raw_ptr,
                                    bool column_major) {
  Tensor handle = FFCObjectWrapper::unwrap(handle_);
  FFModel *model = FFCObjectWrapper::unwrap(model_);
  FFConfig *config = FFCObjectWrapper::unwrap(config_);
  ParallelTensor ptensor;
  model->get_parallel_tensor_from_tensor(handle, ptensor);
  ptensor->attach_raw_ptr(*config, raw_ptr, column_major);
  DEBUG_PRINT("[Tensor] attach numpy array: ptr %p, column_major %d",
              raw_ptr,
              column_major);
}

void flexflow_tensor_detach_raw_ptr(flexflow_tensor_t handle_,
                                    flexflow_model_t model_,
                                    flexflow_config_t config_) {
  Tensor handle = FFCObjectWrapper::unwrap(handle_);
  FFModel *model = FFCObjectWrapper::unwrap(model_);
  FFConfig *config = FFCObjectWrapper::unwrap(config_);
  ParallelTensor ptensor;
  model->get_parallel_tensor_from_tensor(handle, ptensor);
  ptensor->detach_raw_ptr(*config);
}

bool flexflow_tensor_is_mapped(flexflow_tensor_t handle_) {
  assert(false && "Deprecated API");
}

bool flexflow_tensor_set_tensor_float(flexflow_tensor_t handle_,
                                      flexflow_model_t model_,
                                      int num_dim,
                                      int *dims,
                                      float const *data) {
  Tensor handle = FFCObjectWrapper::unwrap(handle_);
  FFModel *model = FFCObjectWrapper::unwrap(model_);
  std::vector<int> dims_vec;
  for (int i = 0; i < num_dim; i++) {
    dims_vec.push_back(dims[i]);
  }
  return handle->set_tensor<float>(model, dims_vec, data);
}

bool flexflow_tensor_get_tensor_float(flexflow_tensor_t handle_,
                                      flexflow_model_t model_,
                                      float *data,
                                      bool get_gradients) {
  Tensor handle = FFCObjectWrapper::unwrap(handle_);
  FFModel *model = FFCObjectWrapper::unwrap(model_);
  return handle->get_tensor<float>(model, data, get_gradients);
}

bool flexflow_tensor_set_tensor_int(flexflow_tensor_t handle_,
                                    flexflow_model_t model_,
                                    int num_dim,
                                    int *dims,
                                    int const *data) {
  Tensor handle = FFCObjectWrapper::unwrap(handle_);
  FFModel *model = FFCObjectWrapper::unwrap(model_);
  std::vector<int> dims_vec;
  for (int i = 0; i < num_dim; i++) {
    dims_vec.push_back(dims[i]);
  }
  return handle->set_tensor<int>(model, dims_vec, data);
}

bool flexflow_tensor_get_tensor_int(flexflow_tensor_t handle_,
                                    flexflow_model_t model_,
                                    int *data,
                                    bool get_gradients) {
  Tensor handle = FFCObjectWrapper::unwrap(handle_);
  FFModel *model = FFCObjectWrapper::unwrap(model_);
  return handle->get_tensor<int>(model, data, get_gradients);
}

bool flexflow_tensor_set_tensor_int64(flexflow_tensor_t handle_,
                                      flexflow_model_t model_,
                                      int num_dim,
                                      int *dims,
                                      int64_t const *data) {
  Tensor handle = FFCObjectWrapper::unwrap(handle_);
  FFModel *model = FFCObjectWrapper::unwrap(model_);
  std::vector<int> dims_vec;
  for (int i = 0; i < num_dim; i++) {
    dims_vec.push_back(dims[i]);
  }
  return handle->set_tensor<int64_t>(model, dims_vec, data);
}

bool flexflow_tensor_get_tensor_int64(flexflow_tensor_t handle_,
                                      flexflow_model_t model_,
                                      int64_t *data,
                                      bool get_gradients) {
  Tensor handle = FFCObjectWrapper::unwrap(handle_);
  FFModel *model = FFCObjectWrapper::unwrap(model_);
  return handle->get_tensor<int64_t>(model, data, get_gradients);
}

bool flexflow_model_get_output_tensor_float(flexflow_model_t model_,
                                            flexflow_tensor_t handle_,
                                            float *data,
                                            bool get_gradients) {
  FFModel *model = FFCObjectWrapper::unwrap(model_);
  Tensor handle = FFCObjectWrapper::unwrap(handle_);
  return handle->get_output_parallel_tensor<float>(model, data, get_gradients);
}

// -----------------------------------------------------------------------
// Parameter
// -----------------------------------------------------------------------

/*
bool
flexflow_parameter_set_weights_float(
  flexflow_parameter_t handle_,
  flexflow_model_t model_,
  int num_dim,
  int *dims,
  const float *data)
{
  Parameter handle = FFCObjectWrapper::unwrap(handle_);
  const FFModel *model = FFCObjectWrapper::unwrap_const(model_);
  std::vector<int> dims_vec;
  for (int i = 0; i < num_dim; i++ ) {
    dims_vec.push_back(dims[i]);
  }
  return handle->set_weights<float>(model, dims_vec, data);
}

bool
flexflow_parameter_get_weights_float(
  flexflow_parameter_t handle_,
  flexflow_model_t model_,
  float *data)
{
  Parameter *handle = FFCObjectWrapper::unwrap(handle_);
  const FFModel *model = FFCObjectWrapper::unwrap_const(model_);
  return handle->get_weights<float>(model, data);
}
*/

// -----------------------------------------------------------------------
// SGDOptimizer
// -----------------------------------------------------------------------

flexflow_sgd_optimizer_t
    flexflow_sgd_optimizer_create(flexflow_model_t model_,
                                  double lr,       /* 0.01f */
                                  double momentum, /* 0.0f */
                                  bool nesterov,   /* false */
                                  double weight_decay /* 0.0f */) {
  FFModel const *model = FFCObjectWrapper::unwrap_const(model_);
  SGDOptimizer *optimizer =
      new SGDOptimizer(model, lr, momentum, nesterov, weight_decay);
  DEBUG_PRINT("[SGDOptimizer] new %p", optimizer);
  return FFCObjectWrapper::wrap(optimizer);
}

void flexflow_sgd_optimizer_destroy(flexflow_sgd_optimizer_t handle_) {
  SGDOptimizer *handle = FFCObjectWrapper::unwrap(handle_);
  DEBUG_PRINT("[SGDOptimizer] delete %p", handle);
  delete handle;
}

void flexflow_sgd_optimizer_set_lr(flexflow_sgd_optimizer_t handle_,
                                   double lr) {
  SGDOptimizer *handle = FFCObjectWrapper::unwrap(handle_);
  handle->lr = lr;
}

// -----------------------------------------------------------------------
// AdamOptimizer
// -----------------------------------------------------------------------

flexflow_adam_optimizer_t
    flexflow_adam_optimizer_create(flexflow_model_t model_,
                                   double alpha /*0.001f*/,
                                   double beta1 /*0.9f*/,
                                   double beta2 /*0.999f*/,
                                   double weight_decay /*0.0f*/,
                                   double epsilon /*1e-8*/) {
  FFModel const *model = FFCObjectWrapper::unwrap_const(model_);
  AdamOptimizer *optimizer =
      new AdamOptimizer(model, alpha, beta1, beta2, weight_decay, epsilon);
  DEBUG_PRINT("AdamOptimizer new %p", optimizer);
  return FFCObjectWrapper::wrap(optimizer);
}

void flexflow_adam_optimizer_destroy(flexflow_adam_optimizer_t handle_) {
  AdamOptimizer *handle = FFCObjectWrapper::unwrap(handle_);
  DEBUG_PRINT("AdamOptimizer delete %p", handle);
  delete handle;
}

void flexflow_adam_optimizer_set_lr(flexflow_adam_optimizer_t handle_,
                                    double lr) {
  AdamOptimizer *handle = FFCObjectWrapper::unwrap(handle_);
  handle->alpha = lr;
}

// -----------------------------------------------------------------------
// Initializer
// -----------------------------------------------------------------------
flexflow_initializer_t flexflow_initializer_create_null() {
  Initializer *initializer = NULL;
  return FFCObjectWrapper::wrap(initializer);
}

// -----------------------------------------------------------------------
// GlorotUniform
// -----------------------------------------------------------------------

flexflow_glorot_uniform_initializer_t
    flexflow_glorot_uniform_initializer_create(int seed) {
  GlorotUniform *initializer = new GlorotUniform(seed);
  DEBUG_PRINT("[GlorotUniform] new %p", initializer);
  return FFCObjectWrapper::wrap(initializer);
}

void flexflow_glorot_uniform_initializer_destroy(
    flexflow_glorot_uniform_initializer_t handle_) {
  GlorotUniform *handle = FFCObjectWrapper::unwrap(handle_);
  DEBUG_PRINT("[GlorotUniform] delete %p", handle);
  delete handle;
}

// -----------------------------------------------------------------------
// ZeroInitializer
// -----------------------------------------------------------------------

flexflow_zero_initializer_t flexflow_zero_initializer_create(void) {
  ZeroInitializer *initializer = new ZeroInitializer();
  DEBUG_PRINT("[ZeroInitializer] new %p", initializer);
  return FFCObjectWrapper::wrap(initializer);
}

void flexflow_zero_initializer_destroy(flexflow_zero_initializer_t handle_) {
  ZeroInitializer *handle = FFCObjectWrapper::unwrap(handle_);
  DEBUG_PRINT("[ZeroInitializer] delete %p", handle);
  delete handle;
}

// -----------------------------------------------------------------------
// UniformInitializer
// -----------------------------------------------------------------------

flexflow_uniform_initializer_t
    flexflow_uniform_initializer_create(int seed, float min, float max) {
  UniformInitializer *initializer = new UniformInitializer(seed, min, max);
  DEBUG_PRINT("[UniformInitializer] new %p", initializer);
  return FFCObjectWrapper::wrap(initializer);
}

void flexflow_uniform_initializer_destroy(
    flexflow_uniform_initializer_t handle_) {
  UniformInitializer *handle = FFCObjectWrapper::unwrap(handle_);
  DEBUG_PRINT("[UniformInitializer] delete %p", handle);
  delete handle;
}

// -----------------------------------------------------------------------
// NormInitializer
// -----------------------------------------------------------------------

flexflow_norm_initializer_t
    flexflow_norm_initializer_create(int seed, float mean, float stddev) {
  NormInitializer *initializer = new NormInitializer(seed, mean, stddev);
  DEBUG_PRINT("[NormInitializer] new %p", initializer);
  return FFCObjectWrapper::wrap(initializer);
}

void flexflow_norm_initializer_destroy(flexflow_norm_initializer_t handle_) {
  NormInitializer *handle = FFCObjectWrapper::unwrap(handle_);
  DEBUG_PRINT("[NormInitializer] delete %p", handle);
  delete handle;
}

// -----------------------------------------------------------------------
// PerfMetrics
// -----------------------------------------------------------------------
void flexflow_per_metrics_destroy(flexflow_perf_metrics_t handle_) {
  PerfMetrics *handle = FFCObjectWrapper::unwrap(handle_);
  delete handle;
  DEBUG_PRINT("[PerfMetrics] delete PerfMetrics %p", handle);
}

float flexflow_per_metrics_get_accuracy(flexflow_perf_metrics_t handle_) {
  PerfMetrics *handle = FFCObjectWrapper::unwrap(handle_);
  float accuracy = handle->train_correct * 100.0f / handle->train_all;
  return accuracy;
}

// -----------------------------------------------------------------------
// NetConfig
// -----------------------------------------------------------------------
flexflow_net_config_t flexflow_net_config_create() {
  NetConfig *netconfig = new NetConfig();
  return FFCObjectWrapper::wrap(netconfig);
}

void flexflow_net_config_destroy(flexflow_net_config_t handle_) {
  NetConfig *handle = FFCObjectWrapper::unwrap(handle_);
  delete handle;
}

char const *
    flexflow_net_config_get_dataset_path(flexflow_net_config_t handle_) {
  NetConfig *handle = FFCObjectWrapper::unwrap(handle_);
  char const *cstr = handle->dataset_path.c_str();
  return cstr;
}

// -----------------------------------------------------------------------
// DLRMConfig
// -----------------------------------------------------------------------
flexflow_dlrm_config_t flexflow_dlrm_config_create() {
  DLRMConfig *netconfig = new DLRMConfig();
  return FFCObjectWrapper::wrap(netconfig);
}

void flexflow_dlrm_config_destroy(flexflow_dlrm_config_t handle_) {
  DLRMConfig *handle = FFCObjectWrapper::unwrap(handle_);
  delete handle;
}

char const *
    flexflow_dlrm_config_get_dataset_path(flexflow_dlrm_config_t handle_) {
  DLRMConfig *handle = FFCObjectWrapper::unwrap(handle_);
  char const *cstr = handle->dataset_path.c_str();
  return cstr;
}

char const *flexflow_dlrm_config_get_arch_interaction_op(
    flexflow_dlrm_config_t handle_) {
  DLRMConfig *handle = FFCObjectWrapper::unwrap(handle_);
  char const *cstr = handle->arch_interaction_op.c_str();
  return cstr;
}

int flexflow_dlrm_config_get_sparse_feature_size(
    flexflow_dlrm_config_t handle_) {
  DLRMConfig *handle = FFCObjectWrapper::unwrap(handle_);
  int result = handle->sparse_feature_size;
  return result;
}

int flexflow_dlrm_config_get_sigmoid_bot(flexflow_dlrm_config_t handle_) {
  DLRMConfig *handle = FFCObjectWrapper::unwrap(handle_);
  int result = handle->sigmoid_bot;
  return result;
}

int flexflow_dlrm_config_get_sigmoid_top(flexflow_dlrm_config_t handle_) {
  DLRMConfig *handle = FFCObjectWrapper::unwrap(handle_);
  int result = handle->sigmoid_top;
  return result;
}

int flexflow_dlrm_config_get_embedding_bag_size(
    flexflow_dlrm_config_t handle_) {
  DLRMConfig *handle = FFCObjectWrapper::unwrap(handle_);
  int result = handle->embedding_bag_size;
  return result;
}

float flexflow_dlrm_config_get_loss_threshold(flexflow_dlrm_config_t handle_) {
  DLRMConfig *handle = FFCObjectWrapper::unwrap(handle_);
  float result = handle->loss_threshold;
  return result;
}

int *flexflow_dlrm_config_get_mlp_bot(flexflow_dlrm_config_t handle_) {
  DLRMConfig *handle = FFCObjectWrapper::unwrap(handle_);
  handle->mlp_bot.insert(handle->mlp_bot.begin(), handle->mlp_bot.size());
  int *result = handle->mlp_bot.data();
  return result;
}

int *flexflow_dlrm_config_get_mlp_top(flexflow_dlrm_config_t handle_) {
  DLRMConfig *handle = FFCObjectWrapper::unwrap(handle_);
  handle->mlp_top.insert(handle->mlp_top.begin(), handle->mlp_top.size());
  int *result = handle->mlp_top.data();
  return result;
}

int *flexflow_dlrm_config_get_embedding_size(flexflow_dlrm_config_t handle_) {
  DLRMConfig *handle = FFCObjectWrapper::unwrap(handle_);
  handle->embedding_size.insert(handle->embedding_size.begin(),
                                handle->embedding_size.size());
  int *result = handle->embedding_size.data();
  return result;
}

// -----------------------------------------------------------------------
// Single Dataloader
// -----------------------------------------------------------------------

flexflow_single_dataloader_t
    flexflow_single_dataloader_create(flexflow_model_t ffmodel_,
                                      flexflow_tensor_t input_,
                                      flexflow_tensor_t full_input_,
                                      int num_samples,
                                      enum DataType data_type) {
  FFModel *ffmodel = FFCObjectWrapper::unwrap(ffmodel_);
  Tensor input = FFCObjectWrapper::unwrap(input_);
  Tensor full_input = FFCObjectWrapper::unwrap(full_input_);
  assert(input->parallel_tensor != nullptr);
  assert(full_input->parallel_tensor != nullptr);
  SingleDataLoader *dataloader =
      new SingleDataLoader(*ffmodel,
                           input->parallel_tensor,
                           full_input->parallel_tensor,
                           num_samples,
                           data_type);
  return FFCObjectWrapper::wrap(dataloader);
}

flexflow_single_dataloader_t
    flexflow_single_dataloader_create2(flexflow_model_t ffmodel_,
                                       flexflow_tensor_t input_,
                                       void *full_input_ptr,
                                       int num_samples,
                                       enum DataType data_type) {
  FFModel *ffmodel = FFCObjectWrapper::unwrap(ffmodel_);
  Tensor input = FFCObjectWrapper::unwrap(input_);
  assert(input->parallel_tensor != nullptr);
  SingleDataLoader *dataloader = new SingleDataLoader(
      *ffmodel, input->parallel_tensor, full_input_ptr, num_samples, data_type);
  return FFCObjectWrapper::wrap(dataloader);
}

void flexflow_single_dataloader_destroy(flexflow_single_dataloader_t handle_) {
  SingleDataLoader *handle = FFCObjectWrapper::unwrap(handle_);
  DEBUG_PRINT("[SingleDataLoader] delete %p", handle);
  delete handle;
}

void flexflow_single_dataloader_set_num_samples(
    flexflow_single_dataloader_t handle_, int samples) {
  SingleDataLoader *handle = FFCObjectWrapper::unwrap(handle_);
  handle->num_samples = samples;
  DEBUG_PRINT("[SingleDataloader] set number of samples %d", samples);
}

int flexflow_single_dataloader_get_num_samples(
    flexflow_single_dataloader_t handle_) {
  SingleDataLoader *handle = FFCObjectWrapper::unwrap(handle_);
  return handle->num_samples;
}

void flexflow_single_dataloader_reset(flexflow_single_dataloader_t handle_) {
  SingleDataLoader *handle = FFCObjectWrapper::unwrap(handle_);
  handle->reset();
}

void flowflow_single_dataloader_next_batch(flexflow_single_dataloader_t handle_,
                                           flexflow_model_t ffmodel_) {
  SingleDataLoader *handle = FFCObjectWrapper::unwrap(handle_);
  FFModel *ffmodel = FFCObjectWrapper::unwrap(ffmodel_);
  handle->next_batch(*ffmodel);
}

// -----------------------------------------------------------------------
// Timer
// -----------------------------------------------------------------------

double flexflow_get_current_time(flexflow_config_t config_) {
  FFConfig *config = FFCObjectWrapper::unwrap(config_);
  config->lg_hlr->issue_execution_fence(config->lg_ctx);
  TimingLauncher timer(MEASURE_MICRO_SECONDS);
  Future future =
      config->lg_hlr->issue_timing_measurement(config->lg_ctx, timer);
  future.get_void_result();
  double ts_start = Realm::Clock::current_time_in_microseconds();
  return ts_start;
}

// -----------------------------------------------------------------------
// Trace
// -----------------------------------------------------------------------

void flexflow_begin_trace(flexflow_config_t config_, int trace_id) {
  FFConfig *config = FFCObjectWrapper::unwrap(config_);
  config->lg_hlr->begin_trace(config->lg_ctx, trace_id);
}

void flexflow_end_trace(flexflow_config_t config_, int trace_id) {
  FFConfig *config = FFCObjectWrapper::unwrap(config_);
  config->lg_hlr->end_trace(config->lg_ctx, trace_id);
}

// -----------------------------------------------------------------------
// Op
// -----------------------------------------------------------------------

int flexflow_op_get_num_parameters(flexflow_op_t handle_) {
  Layer *handle = FFCObjectWrapper::unwrap(handle_);
  return handle->numWeights;
}

flexflow_tensor_t flexflow_op_get_parameter_by_id(flexflow_op_t handle_,
                                                  int id) {
  // assert(false && "TODO: implement a mapping function from parameter to
  // parallel parameter");
  Layer *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor tensor = handle->get_parameter(id);
  return FFCObjectWrapper::wrap(tensor);
}

int flexflow_op_get_num_inputs(flexflow_op_t handle_) {
  Layer *handle = FFCObjectWrapper::unwrap(handle_);
  return handle->numInputs;
}

flexflow_tensor_t flexflow_op_get_input_by_id(flexflow_op_t handle_, int id) {
  Layer *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor tensor = handle->inputs[id];
  return FFCObjectWrapper::wrap(tensor);
}

int flexflow_op_get_num_outputs(flexflow_op_t handle_) {
  Layer *handle = FFCObjectWrapper::unwrap(handle_);
  return handle->numOutputs;
}

flexflow_tensor_t flexflow_op_get_output_by_id(flexflow_op_t handle_, int id) {
  Layer *handle = FFCObjectWrapper::unwrap(handle_);
  Tensor tensor = handle->outputs[id];
  return FFCObjectWrapper::wrap(tensor);
}

void flexflow_op_init(flexflow_op_t handle_, flexflow_model_t model_) {
  assert(false && "Deprecated API");
}

void flexflow_op_forward(flexflow_op_t handle_, flexflow_model_t model_) {
  assert(false && "Deprecated API");
}

// -----------------------------------------------------------------------
// NetConfig implementation
// -----------------------------------------------------------------------
NetConfig::NetConfig(void) {
  InputArgs const &command_args = Runtime::get_input_args();
  char **argv = command_args.argv;
  int argc = command_args.argc;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--dataset")) {
      dataset_path = std::string(argv[++i]);
      continue;
    }
  }
}

// -----------------------------------------------------------------------
// DLRMConfig implementation
// -----------------------------------------------------------------------
DLRMConfig::DLRMConfig(void)
    : sparse_feature_size(2), sigmoid_bot(-1), sigmoid_top(-1),
      embedding_bag_size(1), loss_threshold(0.0f), arch_interaction_op("cat"),
      dataset_path("") {
  embedding_size.push_back(4);
  mlp_bot.push_back(4);
  mlp_bot.push_back(2);
  mlp_top.push_back(8);
  mlp_top.push_back(2);

  InputArgs const &command_args = Runtime::get_input_args();
  char **argv = command_args.argv;
  int argc = command_args.argc;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--arch-sparse-feature-size")) {
      sparse_feature_size = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--arch-embedding-size")) {
      std::stringstream ss((std::string(argv[++i])));
      std::string word;
      embedding_size.clear();
      while (std::getline(ss, word, '-')) {
        embedding_size.push_back(std::stoi(word));
      }
      continue;
    }
    if (!strcmp(argv[i], "--embedding-bag-size")) {
      embedding_bag_size = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--arch-mlp-bot")) {
      std::stringstream ss((std::string(argv[++i])));
      std::string word;
      mlp_bot.clear();
      while (std::getline(ss, word, '-')) {
        mlp_bot.push_back(std::stoi(word));
      }
      continue;
    }
    if (!strcmp(argv[i], "--arch-mlp-top")) {
      std::stringstream ss((std::string(argv[++i])));
      std::string word;
      mlp_top.clear();
      while (std::getline(ss, word, '-')) {
        mlp_top.push_back(std::stoi(word));
      }
      continue;
    }
    if (!strcmp(argv[i], "--loss-threshold")) {
      loss_threshold = atof(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--sigmoid-top")) {
      sigmoid_top = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--sigmoid-bot")) {
      sigmoid_bot = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--arch-interaction-op")) {
      arch_interaction_op = std::string(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--dataset")) {
      dataset_path = std::string(argv[++i]);
      continue;
    }
  }
}

// -----------------------------------------------------------------------
// Registration
// -----------------------------------------------------------------------

void flexflow_registration_callback(Machine machine,
                                    Runtime *runtime,
                                    std::set<Processor> const &local_procs) {
  InputArgs const &command_args = Runtime::get_input_args();
  char **argv = command_args.argv;
  int argc = command_args.argc;
  bool enable_control_replication = true;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--disable-control-replication")) {
      enable_control_replication = false;
      continue;
    }
  }
  register_flexflow_internal_tasks(runtime, false, enable_control_replication);
  SingleDataLoader::register_cpu_tasks(
      runtime, false, enable_control_replication);
  SingleDataLoader::register_gpu_tasks(
      runtime, false, enable_control_replication);
}

void flexflow_perform_registration(void) {
#ifdef FF_USE_NCCL
  // Set NCCL environment
  // This needs to be set, otherwise NCCL will try to use group kernel launches,
  // which are not compatible with the Realm CUDA hijack.
  setenv("NCCL_LAUNCH_MODE", "PARALLEL", true);
#endif
  Runtime::perform_registration_callback(flexflow_registration_callback,
                                         true /*global*/);
  Runtime::perform_registration_callback(FFMapper::update_mappers,
                                         true /*global*/);
}

// -----------------------------------------------------------------------
// BatchConfig
// -----------------------------------------------------------------------

flexflow_batch_config_t flexflow_batch_config_create(void) {
  BatchConfig *config = new BatchConfig();
  DEBUG_PRINT("[BatchConfig] new %p", config);
  return FFCObjectWrapper::wrap(config);
}

void flexflow_batch_config_destroy(flexflow_batch_config_t handle_) {
  BatchConfig *handle = FFCObjectWrapper::unwrap(handle_);
  DEBUG_PRINT("[BatchConfig] delete %p", handle);
  delete handle;
}

// -----------------------------------------------------------------------
// TreeVerifyBatchConfig
// -----------------------------------------------------------------------

flexflow_tree_verify_batch_config_t
    flexflow_tree_verify_batch_config_create(void) {
  TreeVerifyBatchConfig *config = new TreeVerifyBatchConfig();
  DEBUG_PRINT("[TreeVerifyBatchConfig] new %p", config);
  return FFCObjectWrapper::wrap(config);
}

void flexflow_tree_verify_batch_config_destroy(
    flexflow_tree_verify_batch_config_t handle_) {
  TreeVerifyBatchConfig *handle = FFCObjectWrapper::unwrap(handle_);
  DEBUG_PRINT("[TreeVerifyBatchConfig] delete %p", handle);
  delete handle;
}

// -----------------------------------------------------------------------
// BeamSearchBatchConfig
// -----------------------------------------------------------------------

flexflow_beam_search_batch_config_t
    flexflow_beam_search_batch_config_create(void) {
  BeamSearchBatchConfig *config = new BeamSearchBatchConfig();
  DEBUG_PRINT("[BeamSearchBatchConfig] new %p", config);
  return FFCObjectWrapper::wrap(config);
}

void flexflow_beam_search_batch_config_destroy(
    flexflow_beam_search_batch_config_t handle_) {
  BeamSearchBatchConfig *handle = FFCObjectWrapper::unwrap(handle_);
  DEBUG_PRINT("[BeamSearchBatchConfig] delete %p", handle);
  delete handle;
}

// -----------------------------------------------------------------------
// RequestManager
// -----------------------------------------------------------------------

flexflow_request_manager_t flexflow_request_manager_get_request_manager(void) {
  RequestManager *rm = RequestManager::get_request_manager();
  DEBUG_PRINT("[RequestManager] get %p", rm);
  return FFCObjectWrapper::wrap(rm);
}

void flexflow_request_manager_set_max_requests_per_batch(
    flexflow_request_manager_t handle_, int max_num_requests) {
  RequestManager *handle = FFCObjectWrapper::unwrap(handle_);
  handle->set_max_requests_per_batch(max_num_requests);
  DEBUG_PRINT("[RequestManager] set max_requests_per_batch %d",
              max_num_requests);
}

void flexflow_request_manager_set_max_tokens_per_batch(
    flexflow_request_manager_t handle_, int max_num_tokens) {
  RequestManager *handle = FFCObjectWrapper::unwrap(handle_);
  handle->set_max_tokens_per_batch(max_num_tokens);
  DEBUG_PRINT("[RequestManager] set max_tokens_per_batch %d", max_num_tokens);
}

void flexflow_request_manager_set_max_sequence_length(
    flexflow_request_manager_t handle_, int max_seq_length) {
  RequestManager *handle = FFCObjectWrapper::unwrap(handle_);
  handle->set_max_sequence_length(max_seq_length);
  DEBUG_PRINT("[RequestManager] set max_sequence_length %d", max_seq_length);
}

void flexflow_request_manager_register_tokenizer(
    flexflow_request_manager_t handle_,
    enum ModelType model_type,
    int bos_token_id,
    int eos_token_id,
    char const *tokenizer_filepath) {
  RequestManager *handle = FFCObjectWrapper::unwrap(handle_);
  assert(tokenizer_filepath != nullptr &&
         "Cannot convert nullptr char * to std::string");
  std::string const tokenizer_filepath_str(tokenizer_filepath);
  handle->register_tokenizer(
      model_type, bos_token_id, eos_token_id, tokenizer_filepath_str);
  DEBUG_PRINT(
      "[RequestManager] register tokenizer %p %s", handle, tokenizer_filepath);
}

void flexflow_request_manager_register_output_filepath(
    flexflow_request_manager_t handle_, char const *output_filepath) {
  RequestManager *handle = FFCObjectWrapper::unwrap(handle_);
  assert(output_filepath != nullptr &&
         "Cannot convert nullptr char * to std::string");
  std::string const output_filepath_str(output_filepath);
  handle->register_output_filepath(output_filepath_str);
  DEBUG_PRINT("[RequestManager] register output filepath %p %s",
              handle,
              output_filepath);
}

int flexflow_request_manager_register_ssm_model(
    flexflow_request_manager_t handle_, flexflow_model_t model_handle_) {
  RequestManager *handle = FFCObjectWrapper::unwrap(handle_);
  FFModel *model_handle = FFCObjectWrapper::unwrap(model_handle_);
  DEBUG_PRINT("[RequestManager] register ssm %p %p", handle, model_handle);
  return handle->register_ssm_model(model_handle);
}

void flexflow_request_manager_start_background_server(
    flexflow_request_manager_t handle_, flexflow_model_t model_handle_) {
  RequestManager *handle = FFCObjectWrapper::unwrap(handle_);
  FFModel *model_handle = FFCObjectWrapper::unwrap(model_handle_);
  DEBUG_PRINT(
      "[RequestManager] start background server %p %p", handle, model_handle);
  handle->start_background_server(model_handle);
}

void flexflow_request_manager_terminate_background_server(
    flexflow_request_manager_t handle_) {
  RequestManager *handle = FFCObjectWrapper::unwrap(handle_);
  DEBUG_PRINT("[RequestManager] terminate background server %p", handle);
  handle->terminate_background_server();
}

// -----------------------------------------------------------------------
// InferenceManager
// -----------------------------------------------------------------------

flexflow_inference_manager_t
    flexflow_inference_manager_get_inference_manager() {
  InferenceManager *im = InferenceManager::get_inference_manager();
  DEBUG_PRINT("[InferenceManager] get %p", im);
  return FFCObjectWrapper::wrap(im);
}

void flexflow_inference_manager_compile_model_and_allocate_buffer(
    flexflow_inference_manager_t handle_, flexflow_model_t model_handle) {
  InferenceManager *handle = FFCObjectWrapper::unwrap(handle_);
  FFModel *model = FFCObjectWrapper::unwrap(model_handle);
  DEBUG_PRINT("[InferenceManager] compile_model_and_allocate_buffer %p",
              handle);
  handle->compile_model_and_allocate_buffer(model);
}

void flexflow_inference_manager_init_operators_inference(
    flexflow_inference_manager_t handle_, flexflow_model_t model_handle) {
  InferenceManager *handle = FFCObjectWrapper::unwrap(handle_);
  FFModel *model = FFCObjectWrapper::unwrap(model_handle);
  DEBUG_PRINT("[InferenceManager] init_operators_inference %p", handle);
  handle->init_operators_inference(model);
}

void flexflow_inference_manager_register_model_weights_loader(
    flexflow_inference_manager_t handle_,
    flexflow_model_t model_handle,
    flexflow_file_data_loader_t loader_handle) {
  InferenceManager *handle = FFCObjectWrapper::unwrap(handle_);
  FFModel *model = FFCObjectWrapper::unwrap(model_handle);
  FileDataLoader *loader = FFCObjectWrapper::unwrap(loader_handle);
  DEBUG_PRINT("[InferenceManager] register_model_weights_loader %p %p %p",
              handle,
              model,
              loader);
  handle->register_model_weights_loader(model, loader);
}

// -----------------------------------------------------------------------
// FileDataLoader
// -----------------------------------------------------------------------

flexflow_file_data_loader_t
    flexflow_file_data_loader_create(char const *weight_file_path,
                                     int num_q_heads,
                                     int num_kv_heads,
                                     int hidden_dim,
                                     int qkv_inner_dim,
                                     int tensor_parallelism_degree,
                                     bool use_full_precision) {
  assert(weight_file_path != nullptr &&
         "Cannot convert nullptr char * to std::string");
  std::string const weight_file_path_str(weight_file_path);
  FileDataLoader *handle = new FileDataLoader("",
                                              weight_file_path_str,
                                              num_q_heads,
                                              num_kv_heads,
                                              hidden_dim,
                                              qkv_inner_dim,
                                              tensor_parallelism_degree,
                                              use_full_precision);
  DEBUG_PRINT("[FileDataLoader] new %p", handle);
  return FFCObjectWrapper::wrap(handle);
}

void flexflow_file_data_loader_destroy(flexflow_file_data_loader_t handle_) {
  FileDataLoader *handle = FFCObjectWrapper::unwrap(handle_);
  DEBUG_PRINT("[FileDataLoader] delete %p", handle);
  delete handle;
}

void flexflow_file_data_loader_load_weights(flexflow_file_data_loader_t handle_,
                                            flexflow_model_t model_handle_) {
  FileDataLoader *handle = FFCObjectWrapper::unwrap(handle_);
  FFModel *model = FFCObjectWrapper::unwrap(model_handle_);
  handle->load_weights(model);
}

// -----------------------------------------------------------------------
// LoraLinearConfig
// -----------------------------------------------------------------------

flexflow_lora_linear_config_t
    flexflow_lora_linear_config_create(char const *cache_folder_,
                                       char const *peft_model_id_) {
  assert(cache_folder_ != nullptr &&
         "Cannot convert nullptr char * to std::string");
  assert(peft_model_id_ != nullptr &&
         "Cannot convert nullptr char * to std::string");
  std::string const cache_folder(cache_folder_);
  std::string const peft_model_id(peft_model_id_);
  LoraLinearConfig *handle = new LoraLinearConfig(cache_folder, peft_model_id);
  DEBUG_PRINT("[LoraLinearConfig] new %p", handle);
  return FFCObjectWrapper::wrap(handle);
}

void flexflow_lora_linear_config_destroy(
    flexflow_lora_linear_config_t handle_) {
  LoraLinearConfig *peft_config = FFCObjectWrapper::unwrap(handle_);
  DEBUG_PRINT("[LoraLinearConfig] delete %p", peft_config);
  delete peft_config;
}

// -----------------------------------------------------------------------
// PEFTModelID
// -----------------------------------------------------------------------

flexflow_peft_model_id_t flexflow_peft_model_id_create() {
  PEFTModelID *handle = new PEFTModelID();
  DEBUG_PRINT("[PEFTModelID] new %p", handle);
  return FFCObjectWrapper::wrap(handle);
}

flexflow_peft_model_id_t flexflow_peft_model_id_create_id(size_t id) {
  PEFTModelID *handle = new PEFTModelID(id);
  DEBUG_PRINT("[PEFTModelID] new %p", handle);
  return FFCObjectWrapper::wrap(handle);
}

void flexflow_peft_model_id_destroy(flexflow_peft_model_id_t handle_) {
  PEFTModelID *peft_model_id = FFCObjectWrapper::unwrap(handle_);
  DEBUG_PRINT("[PEFTModelID] delete %p", peft_model_id);
  delete peft_model_id;
}
