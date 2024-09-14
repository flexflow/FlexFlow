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
// Inference
FF_NEW_OPAQUE_TYPE(flexflow_batch_config_t);
FF_NEW_OPAQUE_TYPE(flexflow_tree_verify_batch_config_t);
FF_NEW_OPAQUE_TYPE(flexflow_beam_search_batch_config_t);
FF_NEW_OPAQUE_TYPE(flexflow_inference_manager_t);
FF_NEW_OPAQUE_TYPE(flexflow_request_manager_t);
FF_NEW_OPAQUE_TYPE(flexflow_file_data_loader_t);
FF_NEW_OPAQUE_TYPE(flexflow_generation_result_t);

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

int flexflow_config_get_data_parallelism_degree(flexflow_config_t handle_);

int flexflow_config_get_tensor_parallelism_degree(flexflow_config_t handle_);

int flexflow_config_get_pipeline_parallelism_degree(flexflow_config_t handle_);

void flexflow_config_set_data_parallelism_degree(flexflow_config_t handle_,
                                                 int value);

void flexflow_config_set_tensor_parallelism_degree(flexflow_config_t handle_,
                                                   int value);

void flexflow_config_set_pipeline_parallelism_degree(flexflow_config_t handle_,
                                                     int value);

int flexflow_config_get_python_data_loader_type(flexflow_config_t handle);

bool flexflow_config_get_offload(flexflow_config_t handle);

// -----------------------------------------------------------------------
// FFModel
// -----------------------------------------------------------------------

flexflow_model_t flexflow_model_create(flexflow_config_t config,
                                       bool cpu_offload);

void flexflow_model_destroy(flexflow_model_t handle);

void flexflow_model_reset_metrics(flexflow_model_t handle);

void flexflow_model_init_layers(flexflow_model_t handle);

void flexflow_model_prefetch(flexflow_model_t handle);

void flexflow_model_forward(flexflow_model_t handle, int seq_length);

void flexflow_model_backward(flexflow_model_t handle, int seq_length);

void flexflow_model_compute_metrics(flexflow_model_t handle);

void flexflow_model_update(flexflow_model_t handle);

void flexflow_model_compile(flexflow_model_t handle,
                            enum LossType loss_type,
                            int *metrics,
                            int nb_metrics,
                            enum CompMode comp_mode);

flexflow_tensor_t flexflow_model_get_label_tensor(flexflow_model_t handle);

void flexflow_model_zero_gradients(flexflow_model_t handle);

flexflow_tensor_t flexflow_model_add_exp(flexflow_model_t handle,
                                         flexflow_tensor_t const x,
                                         char const *name);

flexflow_tensor_t flexflow_model_add_sin(flexflow_model_t handle,
                                         flexflow_tensor_t const x,
                                         char const *name);

flexflow_tensor_t flexflow_model_add_cos(flexflow_model_t handle,
                                         flexflow_tensor_t const x,
                                         char const *name);

flexflow_tensor_t flexflow_model_add_add(flexflow_model_t handle,
                                         flexflow_tensor_t const x,
                                         flexflow_tensor_t const y,
                                         bool inplace_a,
                                         char const *name);

flexflow_tensor_t flexflow_model_add_subtract(flexflow_model_t handle,
                                              flexflow_tensor_t const x,
                                              flexflow_tensor_t const y,
                                              bool inplace_a,
                                              char const *name);

flexflow_tensor_t flexflow_model_add_multiply(flexflow_model_t handle,
                                              flexflow_tensor_t const x,
                                              flexflow_tensor_t const y,
                                              bool inplace_a,
                                              char const *name);

flexflow_tensor_t flexflow_model_add_divide(flexflow_model_t handle,
                                            flexflow_tensor_t const x,
                                            flexflow_tensor_t const y,
                                            bool inplace_a,
                                            char const *name);

flexflow_tensor_t flexflow_model_add_max(flexflow_model_t handle,
                                         flexflow_tensor_t const x,
                                         flexflow_tensor_t const y,
                                         bool inplace_a,
                                         char const *name);

flexflow_tensor_t flexflow_model_add_min(flexflow_model_t handle,
                                         flexflow_tensor_t const x,
                                         flexflow_tensor_t const y,
                                         bool inplace_a,
                                         char const *name);

flexflow_tensor_t flexflow_model_add_reduce_sum(flexflow_model_t handle_,
                                                flexflow_tensor_t const input_,
                                                int *axes,
                                                int n,
                                                bool keepdims,
                                                char const *name);

flexflow_tensor_t flexflow_model_add_rsqrt(flexflow_model_t handle_,
                                           flexflow_tensor_t const input_,
                                           char const *name);

flexflow_tensor_t flexflow_model_add_pow(flexflow_model_t handle_,
                                         flexflow_tensor_t const input_,
                                         float const exponent,
                                         char const *name);

flexflow_tensor_t flexflow_model_add_mean(flexflow_model_t handle_,
                                          flexflow_tensor_t const input_,
                                          int *dims,
                                          int n,
                                          bool keepdims,
                                          char const *name);

flexflow_tensor_t
    flexflow_model_add_conv2d(flexflow_model_t handle,
                              flexflow_tensor_t const input,
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
                                 flexflow_tensor_t const input,
                                 int num_entries,
                                 int out_dim,
                                 enum AggrMode aggr,
                                 enum DataType dtype,
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
                                                flexflow_tensor_t const input,
                                                bool relu,
                                                char const *name);

flexflow_tensor_t flexflow_model_add_layer_norm(flexflow_model_t handle,
                                                flexflow_tensor_t const input,
                                                int n,
                                                int *axes,
                                                bool elementwise_affine,
                                                float eps,
                                                bool use_bias,
                                                char const *name);

flexflow_tensor_t *
    flexflow_model_add_residual_layer_norm(flexflow_model_t handle,
                                           flexflow_tensor_t const input,
                                           flexflow_tensor_t const residual1,
                                           flexflow_tensor_t const residual2,
                                           bool use_two_residuals,
                                           int n,
                                           int *axes,
                                           bool elementwise_affine,
                                           float eps,
                                           bool use_bias,
                                           char const *name);

flexflow_tensor_t *flexflow_model_add_add_bias_residual_layer_norm(
    flexflow_model_t handle,
    flexflow_tensor_t const input,
    flexflow_tensor_t const residual,
    int n,
    int *axes,
    bool elementwise_affine,
    float eps,
    bool use_bias,
    char const *name);

flexflow_tensor_t
    flexflow_model_add_sigmoid_silu_multi(flexflow_model_t handle,
                                          flexflow_tensor_t const input1,
                                          flexflow_tensor_t const input2,
                                          char const *name);

flexflow_tensor_t
    flexflow_model_add_batch_matmul(flexflow_model_t handle,
                                    flexflow_tensor_t const a,
                                    flexflow_tensor_t const b,
                                    int a_seq_length_dim /* -1 */,
                                    int b_seq_length_dim /* -1 */);

flexflow_tensor_t flexflow_model_add_dense(
    flexflow_model_t handle,
    flexflow_tensor_t const input,
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
                                            flexflow_tensor_t const input,
                                            flexflow_tensor_t const index,
                                            int dim,
                                            char const *name);

flexflow_tensor_t flexflow_model_add_softmax(flexflow_model_t handle,
                                             flexflow_tensor_t const input,
                                             int dim,
                                             char const *name);

flexflow_tensor_t flexflow_model_add_transpose(flexflow_model_t handle,
                                               flexflow_tensor_t const input,
                                               int n,
                                               int *perm,
                                               char const *name);

flexflow_tensor_t flexflow_model_add_reshape(flexflow_model_t handle,
                                             flexflow_tensor_t const input,
                                             int n,
                                             int *shape,
                                             char const *name);

flexflow_tensor_t flexflow_model_add_reverse(flexflow_model_t handle,
                                             flexflow_tensor_t const input,
                                             int axis,
                                             char const *name);

flexflow_tensor_t flexflow_model_add_relu(flexflow_model_t handle,
                                          flexflow_tensor_t const input,
                                          bool inplace,
                                          char const *name);

flexflow_tensor_t
    flexflow_model_add_scalar_multiply(flexflow_model_t handle,
                                       flexflow_tensor_t const input,
                                       float const scalar,
                                       bool inplace,
                                       char const *name);

flexflow_tensor_t flexflow_model_add_scalar_add(flexflow_model_t handle,
                                                flexflow_tensor_t const input,
                                                float const scalar,
                                                bool inplace,
                                                char const *name);

flexflow_tensor_t flexflow_model_add_scalar_sub(flexflow_model_t handle,
                                                flexflow_tensor_t const input,
                                                float const scalar,
                                                bool inplace,
                                                char const *name);

flexflow_tensor_t
    flexflow_model_add_scalar_truediv(flexflow_model_t handle,
                                      flexflow_tensor_t const input,
                                      float const scalar,
                                      bool inplace,
                                      char const *name);

flexflow_tensor_t flexflow_model_add_gelu(flexflow_model_t handle,
                                          flexflow_tensor_t const input,
                                          char const *name);

flexflow_tensor_t flexflow_model_add_identity(flexflow_model_t handle,
                                              flexflow_tensor_t const input,
                                              char const *name);

flexflow_tensor_t flexflow_model_add_sigmoid(flexflow_model_t handle,
                                             flexflow_tensor_t const input,
                                             char const *name);

flexflow_tensor_t flexflow_model_add_tanh(flexflow_model_t handle,
                                          flexflow_tensor_t const input,
                                          char const *name);

flexflow_tensor_t flexflow_model_add_elu(flexflow_model_t handle,
                                         flexflow_tensor_t const input,
                                         bool inplace,
                                         char const *name);

flexflow_tensor_t flexflow_model_add_dropout(flexflow_model_t handle,
                                             flexflow_tensor_t const input,
                                             float rate,
                                             unsigned long long seed,
                                             char const *name);

flexflow_tensor_t flexflow_model_add_multihead_attention(
    flexflow_model_t handle,
    flexflow_tensor_t const query,
    flexflow_tensor_t const key,
    flexflow_tensor_t const value,
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

flexflow_tensor_t flexflow_model_add_inc_multihead_self_attention(
    flexflow_model_t handle_,
    flexflow_tensor_t const input_,
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
    bool streaming_cache,
    char const *name);

flexflow_tensor_t flexflow_model_add_spec_inc_multihead_self_attention(
    flexflow_model_t handle_,
    flexflow_tensor_t const input_,
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
    bool streaming_cache,
    char const *name);

flexflow_tensor_t flexflow_model_add_inc_multihead_self_attention_verify(
    flexflow_model_t handle_,
    flexflow_tensor_t const input_,
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
    char const *name);

flexflow_tensor_t flexflow_model_add_groupquery_self_attention(
    flexflow_model_t handle_,
    flexflow_tensor_t const input_,
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
    bool streaming_cache,
    char const *name);

flexflow_tensor_t flexflow_model_add_spec_inc_multiquery_self_attention(
    flexflow_model_t handle_,
    flexflow_tensor_t const input_,
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
    bool streaming_cache,
    char const *name);

flexflow_tensor_t flexflow_model_add_inc_multiquery_self_attention_verify(
    flexflow_model_t handle_,
    flexflow_tensor_t const input_,
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
    char const *name);

flexflow_tensor_t flexflow_model_add_rms_norm(flexflow_model_t handle_,
                                              flexflow_tensor_t const input_,
                                              float eps,
                                              int dim,
                                              char const *name);

flexflow_tensor_t *
    flexflow_model_add_residual_rms_norm(flexflow_model_t handle_,
                                         flexflow_tensor_t const input1_,
                                         flexflow_tensor_t const input2_,
                                         float eps,
                                         int dim,
                                         char const *name);

flexflow_tensor_t flexflow_model_add_arg_top_k(flexflow_model_t handle_,
                                               flexflow_tensor_t const input_,
                                               int k,
                                               bool sorted,
                                               bool renormalize,
                                               char const *name);

// flexflow_tensor_t flexflow_model_add_beam_top_k(flexflow_model_t handle_,
//                                                 const flexflow_tensor_t
//                                                 input_, int max_beam_size,
//                                                 bool sorted,
//                                                 char const *name);

flexflow_tensor_t flexflow_model_add_sampling(flexflow_model_t handle_,
                                              flexflow_tensor_t const input_,
                                              float top_p,
                                              char const *name);

flexflow_tensor_t flexflow_model_add_argmax(flexflow_model_t handle_,
                                            flexflow_tensor_t const input_,
                                            bool beam_search,
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

void flexflow_model_set_transformer_layer_id(flexflow_model_t handle, int id);

void flexflow_model_generate(flexflow_model_t handle_,
                             int num_requests,
                             char const **input_text,
                             int max_num_chars,
                             char **output_text,
                             int max_seq_length,
                             int **output_length_and_tokens);

void flexflow_model_set_position_offset(flexflow_model_t handle, int offset);

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

// -----------------------------------------------------------------------
// BatchConfig
// -----------------------------------------------------------------------

flexflow_batch_config_t flexflow_batch_config_create(void);

void flexflow_batch_config_destroy(flexflow_batch_config_t handle);

// -----------------------------------------------------------------------
// TreeVerifyBatchConfig
// -----------------------------------------------------------------------

flexflow_tree_verify_batch_config_t
    flexflow_tree_verify_batch_config_create(void);

void flexflow_tree_verify_batch_config_destroy(
    flexflow_tree_verify_batch_config_t handle);

// -----------------------------------------------------------------------
// BeamSearchBatchConfig
// -----------------------------------------------------------------------

flexflow_beam_search_batch_config_t
    flexflow_beam_search_batch_config_create(void);

void flexflow_beam_search_batch_config_destroy(
    flexflow_beam_search_batch_config_t handle);

// -----------------------------------------------------------------------
// RequestManager
// -----------------------------------------------------------------------

flexflow_request_manager_t flexflow_request_manager_get_request_manager(void);

// void flexflow_request_manager_destroy(flexflow_request_manager_t handle_);

void flexflow_request_manager_set_max_requests_per_batch(
    flexflow_request_manager_t handle_, int max_num_requests);

void flexflow_request_manager_set_max_tokens_per_batch(
    flexflow_request_manager_t handle_, int max_num_tokens);

void flexflow_request_manager_set_max_tokens_per_ssm_batch(
    flexflow_request_manager_t handle_, int max_num_ssm_tokens);

void flexflow_request_manager_set_max_tokens_per_prefilling_batch(
    flexflow_request_manager_t handle_, int max_num_prefilling_tokens);

void flexflow_request_manager_set_max_sequence_length(
    flexflow_request_manager_t handle_, int max_seq_length);

void flexflow_request_manager_register_tokenizer(
    flexflow_request_manager_t handle_,
    enum ModelType model_type,
    int bos_token_id,
    int eos_token_id,
    char const *tokenizer_filepath);

void flexflow_request_manager_register_output_filepath(
    flexflow_request_manager_t handle_, char const *output_filepath);

int flexflow_request_manager_register_ssm_model(
    flexflow_request_manager_t handle_, flexflow_model_t model_handle_);

void flexflow_request_manager_start_background_server(
    flexflow_request_manager_t handle_, flexflow_model_t model_handle_);

void flexflow_request_manager_terminate_background_server(
    flexflow_request_manager_t handle_);

// -----------------------------------------------------------------------
// InferenceManager
// -----------------------------------------------------------------------

flexflow_inference_manager_t
    flexflow_inference_manager_get_inference_manager(void);

// void flexflow_inference_manager_destroy(flexflow_inference_manager_t
// handle_);

void flexflow_inference_manager_compile_model_and_allocate_buffer(
    flexflow_inference_manager_t handle_, flexflow_model_t model_handle);

void flexflow_inference_manager_init_operators_inference(
    flexflow_inference_manager_t handle_, flexflow_model_t model_handle);

void flexflow_inference_manager_register_model_weights_loader(
    flexflow_inference_manager_t handle_,
    flexflow_model_t model_handle,
    flexflow_file_data_loader_t loader_handle);

// -----------------------------------------------------------------------
// FileDataLoader
// -----------------------------------------------------------------------

flexflow_file_data_loader_t
    flexflow_file_data_loader_create(char const *weight_file_path,
                                     int num_q_heads,
                                     int num_kv_heads,
                                     int hidden_dim,
                                     int head_dim,
                                     int tensor_parallelism_degree,
                                     bool use_full_precision);

void flexflow_file_data_loader_destroy(flexflow_file_data_loader_t handle_);

void flexflow_file_data_loader_load_weights(flexflow_file_data_loader_t handle_,
                                            flexflow_model_t model_handle_);

#ifdef __cplusplus
}
#endif

#endif // __FLEXFLOW_C_H__
