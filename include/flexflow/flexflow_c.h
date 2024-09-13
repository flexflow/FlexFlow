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

#ifndef __FLEXFLOW_C_H__
#define __FLEXFLOW_C_H__

#include "flexflow/ffconst.h"
#include "stdint.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FF_NEW_OPAQUE_TYPE(T)                                                  \
  typedef struct T {                                                           \
    void *impl;                                                                \
  } T

FF_NEW_OPAQUE_TYPE(flexflow_config_t);
FF_NEW_OPAQUE_TYPE(flexflow_model_t);
FF_NEW_OPAQUE_TYPE(flexflow_tensor_t);
FF_NEW_OPAQUE_TYPE(flexflow_parallel_tensor_t);
FF_NEW_OPAQUE_TYPE(flexflow_sgd_optimizer_t);
FF_NEW_OPAQUE_TYPE(flexflow_adam_optimizer_t);
FF_NEW_OPAQUE_TYPE(flexflow_initializer_t);
FF_NEW_OPAQUE_TYPE(flexflow_glorot_uniform_initializer_t);
FF_NEW_OPAQUE_TYPE(flexflow_zero_initializer_t);
FF_NEW_OPAQUE_TYPE(flexflow_uniform_initializer_t);
FF_NEW_OPAQUE_TYPE(flexflow_norm_initializer_t);
FF_NEW_OPAQUE_TYPE(flexflow_op_t);
// FF_NEW_OPAQUE_TYPE(flexflow_parameter_t);
FF_NEW_OPAQUE_TYPE(flexflow_perf_metrics_t);
FF_NEW_OPAQUE_TYPE(flexflow_net_config_t);
FF_NEW_OPAQUE_TYPE(flexflow_dlrm_config_t);
FF_NEW_OPAQUE_TYPE(flexflow_dataloader_4d_t);
FF_NEW_OPAQUE_TYPE(flexflow_dataloader_2d_t);
FF_NEW_OPAQUE_TYPE(flexflow_single_dataloader_t);

// -----------------------------------------------------------------------
// FFConfig
// -----------------------------------------------------------------------

flexflow_config_t flexflow_config_create(void);

void flexflow_config_destroy(flexflow_config_t handle);

void flexflow_config_parse_args(flexflow_config_t handle,
                                char **argv,
                                int argc);

void flexflow_config_parse_args_default(flexflow_config_t handle);

int flexflow_config_get_batch_size(flexflow_config_t handle);

int flexflow_config_get_workers_per_node(flexflow_config_t handle);

int flexflow_config_get_num_nodes(flexflow_config_t handle);

int flexflow_config_get_epochs(flexflow_config_t handle);

bool flexflow_config_get_enable_control_replication(flexflow_config_t handle);

int flexflow_config_get_python_data_loader_type(flexflow_config_t handle);
// -----------------------------------------------------------------------
// FFModel
// -----------------------------------------------------------------------

flexflow_model_t flexflow_model_create(flexflow_config_t config);

void flexflow_model_destroy(flexflow_model_t handle);

void flexflow_model_reset_metrics(flexflow_model_t handle);

void flexflow_model_init_layers(flexflow_model_t handle);

void flexflow_model_prefetch(flexflow_model_t handle);

void flexflow_model_forward(flexflow_model_t handle, int seq_length);

void flexflow_model_backward(flexflow_model_t handle, int seq_length);

void flexflow_model_compute_metrics(flexflow_model_t handle);

void flexflow_model_update(flexflow_model_t handle);

void flexflow_model_unified_update(flexflow_model_t handle);

void flexflow_model_compile(flexflow_model_t handle,
                            enum LossType loss_type,
                            int *metrics,
                            int nb_metrics,
                            enum CompMode comp_mode);

flexflow_tensor_t flexflow_model_get_label_tensor(flexflow_model_t handle);

void flexflow_model_zero_gradients(flexflow_model_t handle);

flexflow_tensor_t flexflow_model_add_exp(flexflow_model_t handle,
                                         const flexflow_tensor_t x,
                                         char const *name);

flexflow_tensor_t flexflow_model_add_sin(flexflow_model_t handle,
                                         const flexflow_tensor_t x,
                                         char const *name);

flexflow_tensor_t flexflow_model_add_cos(flexflow_model_t handle,
                                         const flexflow_tensor_t x,
                                         char const *name);

flexflow_tensor_t flexflow_model_add_add(flexflow_model_t handle,
                                         const flexflow_tensor_t x,
                                         const flexflow_tensor_t y,
                                         bool inplace_a,
                                         char const *name);

flexflow_tensor_t flexflow_model_add_subtract(flexflow_model_t handle,
                                              const flexflow_tensor_t x,
                                              const flexflow_tensor_t y,
                                              bool inplace_a,
                                              char const *name);

flexflow_tensor_t flexflow_model_add_multiply(flexflow_model_t handle,
                                              const flexflow_tensor_t x,
                                              const flexflow_tensor_t y,
                                              bool inplace_a,
                                              char const *name);

flexflow_tensor_t flexflow_model_add_divide(flexflow_model_t handle,
                                            const flexflow_tensor_t x,
                                            const flexflow_tensor_t y,
                                            bool inplace_a,
                                            char const *name);

flexflow_tensor_t flexflow_model_add_max(flexflow_model_t handle,
                                         const flexflow_tensor_t x,
                                         const flexflow_tensor_t y,
                                         bool inplace_a,
                                         char const *name);

flexflow_tensor_t flexflow_model_add_min(flexflow_model_t handle,
                                         const flexflow_tensor_t x,
                                         const flexflow_tensor_t y,
                                         bool inplace_a,
                                         char const *name);

flexflow_tensor_t flexflow_model_add_reduce_sum(flexflow_model_t handle_,
                                                const flexflow_tensor_t input_,
                                                int *axes,
                                                int n,
                                                bool keepdims,
                                                char const *name);

flexflow_tensor_t flexflow_model_add_rsqrt(flexflow_model_t handle_,
                                           const flexflow_tensor_t input_,
                                           char const *name);

flexflow_tensor_t flexflow_model_add_pow(flexflow_model_t handle_,
                                         const flexflow_tensor_t input_,
                                         float const exponent,
                                         char const *name);

flexflow_tensor_t flexflow_model_add_mean(flexflow_model_t handle_,
                                          const flexflow_tensor_t input_,
                                          int *dims,
                                          int n,
                                          bool keepdims,
                                          char const *name);

flexflow_tensor_t
    flexflow_model_add_conv2d(flexflow_model_t handle,
                              const flexflow_tensor_t input,
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
                              flexflow_op_t shared_op,
                              flexflow_initializer_t kernel_initializer,
                              flexflow_initializer_t bias_initializer,
                              char const *name);

flexflow_tensor_t
    flexflow_model_add_embedding(flexflow_model_t handle,
                                 const flexflow_tensor_t input,
                                 int num_entires,
                                 int out_dim,
                                 enum AggrMode aggr,
                                 flexflow_op_t shared_op,
                                 flexflow_initializer_t kernel_initializer,
                                 char const *name);

flexflow_tensor_t
    flexflow_model_add_pool2d(flexflow_model_t handle,
                              flexflow_tensor_t input,
                              int kernel_h,
                              int kernel_w,
                              int stride_h,
                              int stride_w,
                              int padding_h,
                              int padding_w,
                              enum PoolType type /* POOL_MAX */,
                              enum ActiMode activation /* AC_MODE_NONE */,
                              char const *name);

flexflow_tensor_t flexflow_model_add_batch_norm(flexflow_model_t handle,
                                                const flexflow_tensor_t input,
                                                bool relu,
                                                char const *name);

flexflow_tensor_t flexflow_model_add_layer_norm(flexflow_model_t handle,
                                                const flexflow_tensor_t input,
                                                int n,
                                                int *axes,
                                                bool elementwise_affine,
                                                float eps,
                                                char const *name);

flexflow_tensor_t
    flexflow_model_add_batch_matmul(flexflow_model_t handle,
                                    const flexflow_tensor_t a,
                                    const flexflow_tensor_t b,
                                    int a_seq_length_dim /* -1 */,
                                    int b_seq_length_dim /* -1 */);

flexflow_tensor_t flexflow_model_add_dense(
    flexflow_model_t handle,
    const flexflow_tensor_t input,
    int out_dim,
    enum ActiMode activation /* AC_MODE_NONE */,
    bool use_bias /* true */,
    enum DataType data_type /*DT_FLOAT*/,
    flexflow_op_t shared_op,
    flexflow_initializer_t kernel_initializer,
    flexflow_initializer_t bias_initializer,
    enum RegularizerMode kernel_reg_type /* REG_MODE_NONE */,
    float kernel_reg_lambda,
    char const *name);

flexflow_tensor_t flexflow_model_add_concat(flexflow_model_t handle,
                                            int n,
                                            flexflow_tensor_t *input,
                                            int axis,
                                            char const *name);

void flexflow_model_add_split(flexflow_model_t handle,
                              flexflow_tensor_t input,
                              int n,
                              flexflow_tensor_t *outputs,
                              int *split,
                              int axis,
                              char const *name);

flexflow_tensor_t flexflow_model_add_flat(flexflow_model_t handle,
                                          flexflow_tensor_t input,
                                          char const *name);

flexflow_tensor_t flexflow_model_add_gather(flexflow_model_t handle,
                                            const flexflow_tensor_t input,
                                            const flexflow_tensor_t index,
                                            int dim,
                                            char const *name);

flexflow_tensor_t flexflow_model_add_softmax(flexflow_model_t handle,
                                             const flexflow_tensor_t input,
                                             int dim,
                                             bool last_layer,
                                             char const *name);

flexflow_tensor_t flexflow_model_add_transpose(flexflow_model_t handle,
                                               const flexflow_tensor_t input,
                                               int n,
                                               int *perm,
                                               char const *name);

flexflow_tensor_t flexflow_model_add_reshape(flexflow_model_t handle,
                                             const flexflow_tensor_t input,
                                             int n,
                                             int *shape,
                                             char const *name);

flexflow_tensor_t flexflow_model_add_reverse(flexflow_model_t handle,
                                             const flexflow_tensor_t input,
                                             int axis,
                                             char const *name);

flexflow_tensor_t flexflow_model_add_relu(flexflow_model_t handle,
                                          const flexflow_tensor_t input,
                                          bool inplace,
                                          char const *name);

flexflow_tensor_t
    flexflow_model_add_scalar_multiply(flexflow_model_t handle,
                                       const flexflow_tensor_t input,
                                       float const scalar,
                                       bool inplace,
                                       char const *name);

flexflow_tensor_t flexflow_model_add_scalar_add(flexflow_model_t handle,
                                                const flexflow_tensor_t input,
                                                float const scalar,
                                                bool inplace,
                                                char const *name);

flexflow_tensor_t flexflow_model_add_scalar_sub(flexflow_model_t handle,
                                                const flexflow_tensor_t input,
                                                float const scalar,
                                                bool inplace,
                                                char const *name);

flexflow_tensor_t
    flexflow_model_add_scalar_truediv(flexflow_model_t handle,
                                      const flexflow_tensor_t input,
                                      float const scalar,
                                      bool inplace,
                                      char const *name);

flexflow_tensor_t flexflow_model_add_gelu(flexflow_model_t handle,
                                          const flexflow_tensor_t input,
                                          char const *name);

flexflow_tensor_t flexflow_model_add_identity(flexflow_model_t handle,
                                              const flexflow_tensor_t input,
                                              char const *name);

flexflow_tensor_t flexflow_model_add_sigmoid(flexflow_model_t handle,
                                             const flexflow_tensor_t input,
                                             char const *name);

flexflow_tensor_t flexflow_model_add_tanh(flexflow_model_t handle,
                                          const flexflow_tensor_t input,
                                          char const *name);

flexflow_tensor_t flexflow_model_add_elu(flexflow_model_t handle,
                                         const flexflow_tensor_t input,
                                         bool inplace,
                                         char const *name);

flexflow_tensor_t flexflow_model_add_dropout(flexflow_model_t handle,
                                             const flexflow_tensor_t input,
                                             float rate,
                                             unsigned long long seed,
                                             char const *name);

flexflow_tensor_t flexflow_model_add_multihead_attention(
    flexflow_model_t handle,
    const flexflow_tensor_t query,
    const flexflow_tensor_t key,
    const flexflow_tensor_t value,
    int embed_dim,
    int num_heads,
    int kdim,
    int vdim,
    float dropout,
    bool bias,
    bool add_bias_kv,
    bool add_zero_attn,
    flexflow_initializer_t kernel_initializer,
    char const *name);

void flexflow_model_set_sgd_optimizer(flexflow_model_t handle,
                                      flexflow_sgd_optimizer_t optimizer);

void flexflow_model_set_adam_optimizer(flexflow_model_t handle,
                                       flexflow_adam_optimizer_t optimizer);

void flexflow_model_print_layers(flexflow_model_t handle, int id);

flexflow_op_t flexflow_model_get_layer_by_id(flexflow_model_t handle,
                                             int layer_id);

flexflow_op_t flexflow_model_get_last_layer(flexflow_model_t handle);

flexflow_tensor_t flexflow_model_get_parameter_by_id(flexflow_model_t handle,
                                                     int layer_id);

flexflow_perf_metrics_t
    flexflow_model_get_perf_metrics(flexflow_model_t handle);

// -----------------------------------------------------------------------
// Tensor
// -----------------------------------------------------------------------

flexflow_tensor_t flexflow_tensor_create(flexflow_model_t model,
                                         int num_dims,
                                         int const *dims,
                                         enum DataType data_type,
                                         bool create_grad /* true */);

void flexflow_tensor_map(flexflow_model_t model,
                         flexflow_tensor_t tensor,
                         flexflow_op_t op);

flexflow_tensor_t flexflow_constant_create(flexflow_model_t model,
                                           int num_dims,
                                           int const *dims,
                                           float value,
                                           enum DataType data_type);

void flexflow_tensor_destroy(flexflow_tensor_t handle);

void flexflow_tensor_inline_map(flexflow_tensor_t handle,
                                flexflow_model_t model,
                                flexflow_config_t config);

void flexflow_tensor_inline_unmap(flexflow_tensor_t handle,
                                  flexflow_model_t model,
                                  flexflow_config_t config);

float *flexflow_tensor_get_raw_ptr_float(flexflow_tensor_t handle,
                                         flexflow_model_t model,
                                         flexflow_config_t config);

int32_t *flexflow_tensor_get_raw_ptr_int32(flexflow_tensor_t handle,
                                           flexflow_model_t model,
                                           flexflow_config_t config);

int flexflow_tensor_get_num_dims(flexflow_tensor_t handle);

int flexflow_tensor_get_dim(flexflow_tensor_t handle, int legion_axis);

int *flexflow_tensor_get_dims(flexflow_tensor_t handle);

int flexflow_tensor_get_data_type(flexflow_tensor_t handle);

flexflow_op_t flexflow_tensor_get_owner_op(flexflow_tensor_t handle);

void flexflow_tensor_attach_raw_ptr(flexflow_tensor_t handle,
                                    flexflow_model_t model,
                                    flexflow_config_t config,
                                    void *raw_ptr,
                                    bool column_major);

void flexflow_tensor_detach_raw_ptr(flexflow_tensor_t handle,
                                    flexflow_model_t model,
                                    flexflow_config_t config);

bool flexflow_tensor_is_mapped(flexflow_tensor_t handle);

bool flexflow_tensor_set_tensor_float(flexflow_tensor_t handle,
                                      flexflow_model_t model,
                                      int num_dim,
                                      int *dims,
                                      float const *data);

bool flexflow_tensor_get_tensor_float(flexflow_tensor_t handle,
                                      flexflow_model_t model,
                                      float *data,
                                      bool get_gradients);

bool flexflow_tensor_set_tensor_int(flexflow_tensor_t handle,
                                    flexflow_model_t model,
                                    int num_dim,
                                    int *dims,
                                    int const *data);

bool flexflow_tensor_get_tensor_int(flexflow_tensor_t handle,
                                    flexflow_model_t model,
                                    int *data,
                                    bool get_gradients);

bool flexflow_tensor_set_tensor_int64(flexflow_tensor_t handle,
                                      flexflow_model_t model,
                                      int num_dim,
                                      int *dims,
                                      int64_t const *data,
                                      enum ParameterSyncType comm_type);

bool flexflow_tensor_get_tensor_int64(flexflow_tensor_t handle,
                                      flexflow_model_t model,
                                      int64_t *data,
                                      bool get_gradients);

bool flexflow_model_get_output_tensor_float(flexflow_model_t model_,
                                            flexflow_tensor_t handle_,
                                            float *data,
                                            bool get_gradients);

// -----------------------------------------------------------------------
// Parameter
// -----------------------------------------------------------------------
/*
bool
flexflow_parameter_set_weights_float(
  flexflow_tensor_t handle,
  flexflow_model_t model,
  int num_dim,
  int *dims,
  const float *data);

bool
flexflow_parameter_get_weights_float(
  flexflow_parameter_t handle,
  flexflow_model_t model,
  float *data);
*/
// -----------------------------------------------------------------------
// SGDOptimizer
// -----------------------------------------------------------------------

flexflow_sgd_optimizer_t
    flexflow_sgd_optimizer_create(flexflow_model_t model,
                                  double lr,       /* 0.01f */
                                  double momentum, /* 0.0f */
                                  bool nesterov,   /* false */
                                  double weight_decay /* 0.0f */);

void flexflow_sgd_optimizer_destroy(flexflow_sgd_optimizer_t handle);

void flexflow_sgd_optimizer_set_lr(flexflow_sgd_optimizer_t handle, double lr);

// -----------------------------------------------------------------------
// AdamOptimizer
// -----------------------------------------------------------------------

flexflow_adam_optimizer_t
    flexflow_adam_optimizer_create(flexflow_model_t model,
                                   double alpha /*0.001f*/,
                                   double beta1 /*0.9f*/,
                                   double beta2 /*0.999f*/,
                                   double weight_decay /*0.0f*/,
                                   double epsilon /*1e-8*/);

void flexflow_adam_optimizer_destroy(flexflow_adam_optimizer_t handle);

void flexflow_adam_optimizer_set_lr(flexflow_adam_optimizer_t handle,
                                    double lr);

// -----------------------------------------------------------------------
// Initializer
// -----------------------------------------------------------------------
flexflow_initializer_t flexflow_initializer_create_null();

// -----------------------------------------------------------------------
// GlorotUniform
// -----------------------------------------------------------------------

flexflow_glorot_uniform_initializer_t
    flexflow_glorot_uniform_initializer_create(int seed);

void flexflow_glorot_uniform_initializer_destroy(
    flexflow_glorot_uniform_initializer_t handle);

// -----------------------------------------------------------------------
// ZeroInitializer
// -----------------------------------------------------------------------

flexflow_zero_initializer_t flexflow_zero_initializer_create(void);

void flexflow_zero_initializer_destroy(flexflow_zero_initializer_t handle);

// -----------------------------------------------------------------------
// UniformInitializer
// -----------------------------------------------------------------------

flexflow_uniform_initializer_t
    flexflow_uniform_initializer_create(int seed, float min, float max);

void flexflow_uniform_initializer_destroy(
    flexflow_uniform_initializer_t handle);

// -----------------------------------------------------------------------
// NormInitializer
// -----------------------------------------------------------------------

flexflow_norm_initializer_t
    flexflow_norm_initializer_create(int seed, float mean, float stddev);

void flexflow_norm_initializer_destroy(flexflow_norm_initializer_t handle);

// -----------------------------------------------------------------------
// PerfMetrics
// -----------------------------------------------------------------------
void flexflow_per_metrics_destroy(flexflow_perf_metrics_t handle);

float flexflow_per_metrics_get_accuracy(flexflow_perf_metrics_t handle);

// -----------------------------------------------------------------------
// NetConfig
// -----------------------------------------------------------------------

flexflow_net_config_t flexflow_net_config_create();

void flexflow_net_config_destroy(flexflow_net_config_t handle);

char const *flexflow_net_config_get_dataset_path(flexflow_net_config_t handle);

// -----------------------------------------------------------------------
// DLRMConfig
// -----------------------------------------------------------------------

flexflow_dlrm_config_t flexflow_dlrm_config_create();

void flexflow_dlrm_config_destroy(flexflow_dlrm_config_t handle);

char const *
    flexflow_dlrm_config_get_dataset_path(flexflow_dlrm_config_t handle);

char const *
    flexflow_dlrm_config_get_arch_interaction_op(flexflow_dlrm_config_t handle);

int flexflow_dlrm_config_get_sparse_feature_size(flexflow_dlrm_config_t handle);

int flexflow_dlrm_config_get_sigmoid_bot(flexflow_dlrm_config_t handle);

int flexflow_dlrm_config_get_sigmoid_top(flexflow_dlrm_config_t handle);

int flexflow_dlrm_config_get_embedding_bag_size(flexflow_dlrm_config_t handle);

float flexflow_dlrm_config_get_loss_threshold(flexflow_dlrm_config_t handle);

int *flexflow_dlrm_config_get_mlp_bot(flexflow_dlrm_config_t handle);

int *flexflow_dlrm_config_get_mlp_top(flexflow_dlrm_config_t handle);

int *flexflow_dlrm_config_get_embedding_size(flexflow_dlrm_config_t handle);

// -----------------------------------------------------------------------
// Single Dataloader
// -----------------------------------------------------------------------

flexflow_single_dataloader_t
    flexflow_single_dataloader_create(flexflow_model_t ffmodel,
                                      flexflow_tensor_t input,
                                      flexflow_tensor_t full_input,
                                      int num_samples,
                                      enum DataType data_type);

flexflow_single_dataloader_t
    flexflow_single_dataloader_create2(flexflow_model_t ffmodel,
                                       flexflow_tensor_t input,
                                       void *full_input_ptr,
                                       int num_samples,
                                       enum DataType data_type);

void flexflow_single_dataloader_destroy(flexflow_single_dataloader_t handle);

void flexflow_single_dataloader_set_num_samples(
    flexflow_single_dataloader_t handle, int samples);

int flexflow_single_dataloader_get_num_samples(
    flexflow_single_dataloader_t handle);

void flexflow_single_dataloader_reset(flexflow_single_dataloader_t handle);

void flowflow_single_dataloader_next_batch(flexflow_single_dataloader_t handle,
                                           flexflow_model_t ffmodel);

// -----------------------------------------------------------------------
// Timer
// -----------------------------------------------------------------------

double flexflow_get_current_time(flexflow_config_t config);

// -----------------------------------------------------------------------
// Trace
// -----------------------------------------------------------------------

void flexflow_begin_trace(flexflow_config_t config, int trace_id);

void flexflow_end_trace(flexflow_config_t config, int trace_id);

// -----------------------------------------------------------------------
// Op
// -----------------------------------------------------------------------

int flexflow_op_get_num_parameters(flexflow_op_t handle);

flexflow_tensor_t flexflow_op_get_parameter_by_id(flexflow_op_t handle, int id);

int flexflow_op_get_num_inputs(flexflow_op_t handle);

flexflow_tensor_t flexflow_op_get_input_by_id(flexflow_op_t handle, int id);

int flexflow_op_get_num_outputs(flexflow_op_t handle);

flexflow_tensor_t flexflow_op_get_output_by_id(flexflow_op_t handle, int id);

void flexflow_op_init(flexflow_op_t handle, flexflow_model_t model);

void flexflow_op_forward(flexflow_op_t handle, flexflow_model_t model);

// -----------------------------------------------------------------------
// Registration
// -----------------------------------------------------------------------

void flexflow_perform_registration(void);

#ifdef __cplusplus
}
#endif

#endif // __FLEXFLOW_C_H__
