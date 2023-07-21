#ifndef _FLEXFLOW_PCG_FFI_INCLUDE_FLEXFLOW_PCG_H
#define _FLEXFLOW_PCG_FFI_INCLUDE_FLEXFLOW_PCG_H

#include "flexflow/op-attrs.h"
#include "flexflow/utils.h"
#include <stddef.h>
#include <stdio.h>

FLEXFLOW_FFI_BEGIN();

typedef enum {
  FLEXFLOW_PCG_STATUS_OK,
  FLEXFLOW_PCG_ERROR_UNKNOWN,
} flexflow_pcg_error_t;

typedef enum {
  FLEXFLOW_DEVICE_TYPE_CPU,
  FLEXFLOW_DEVICE_TYPE_GPU,
} flexflow_device_type_t;

FF_NEW_OPAQUE_TYPE(flexflow_computation_graph_t);
FF_NEW_OPAQUE_TYPE(flexflow_parallel_computation_graph_t);
FF_NEW_OPAQUE_TYPE(flexflow_operator_t);
FF_NEW_OPAQUE_TYPE(flexflow_parallel_tensor_t);
FF_NEW_OPAQUE_TYPE(flexflow_layer_t);
FF_NEW_OPAQUE_TYPE(flexflow_tensor_t);
FF_NEW_OPAQUE_TYPE(flexflow_machine_view_t);
FF_NEW_OPAQUE_TYPE(flexflow_initializer_t);
FF_NEW_OPAQUE_TYPE(flexflow_optimizer_t);
FF_NEW_OPAQUE_TYPE(flexflow_machine_specification_t);
FF_NEW_OPAQUE_TYPE(flexflow_model_compilation_input_t);
FF_NEW_OPAQUE_TYPE(flexflow_model_compilation_result_t);

extern flexflow_initializer_t NO_INITIALIZER;

flexflow_error_t make_pcg_error(flexflow_pcg_error_t);

char *flexflow_pcg_get_error_string(flexflow_pcg_error_t);

flexflow_error_t
    flexflow_computation_graph_create(flexflow_computation_graph_t *out);
flexflow_error_t
    flexflow_computation_graph_destroy(flexflow_computation_graph_t);

flexflow_error_t flexflow_computation_graph_serialize_to_buf(
    flexflow_computation_graph_t, void *buf, size_t buf_size);
flexflow_error_t flexflow_computation_graph_deserialize_from_buf(
    flexflow_computation_graph_t, void *buf, size_t buf_size);

flexflow_error_t
    flexflow_computation_graph_serialize_to_file(flexflow_computation_graph_t,
                                                 FILE *);
flexflow_error_t flexflow_computation_graph_deserialize_from_file(
    FILE *, flexflow_computation_graph_t *);

flexflow_error_t flexflow_tensor_create(flexflow_computation_graph_t,
                                        int num_dims,
                                        int *dims,
                                        flexflow_datatype_t datatype,
                                        bool create_grad,
                                        flexflow_tensor_t *out);
flexflow_error_t flexflow_tensor_get_create_grad(flexflow_tensor_t, bool *out);
flexflow_error_t flexflow_tensor_get_initializer(flexflow_tensor_t,
                                                 flexflow_initializer_t *out);
flexflow_error_t flexflow_tensor_get_sync_type(flexflow_tensor_t,
                                               flexflow_param_sync_t *out);
flexflow_error_t flexflow_tensor_get_datatype(flexflow_tensor_t,
                                              flexflow_datatype_t *out);
flexflow_error_t flexflow_tensor_get_num_dims(flexflow_tensor_t, int *out);
flexflow_error_t flexflow_tensor_get_dims(flexflow_tensor_t, int *out);
flexflow_error_t flexflow_tensor_destroy(flexflow_computation_graph_t,
                                         flexflow_tensor_t);

flexflow_error_t
    flexflow_computation_graph_add_op_exp(flexflow_computation_graph_t,
                                          flexflow_tensor_t,
                                          flexflow_tensor_t *out,
                                          char *name = NULL);
flexflow_error_t
    flexflow_computation_graph_add_op_add(flexflow_computation_graph_t,
                                          flexflow_tensor_t,
                                          flexflow_tensor_t,
                                          flexflow_tensor_t *out,
                                          char *name = NULL);
flexflow_error_t
    flexflow_computation_graph_add_op_subtract(flexflow_computation_graph_t,
                                               flexflow_tensor_t,
                                               flexflow_tensor_t,
                                               flexflow_tensor_t *out,
                                               char *name = NULL);
flexflow_error_t
    flexflow_computation_graph_add_op_multiply(flexflow_computation_graph_t,
                                               flexflow_tensor_t,
                                               flexflow_tensor_t,
                                               flexflow_tensor_t *out,
                                               char *name = NULL);
flexflow_error_t
    flexflow_computation_graph_add_op_divide(flexflow_computation_graph_t,
                                             flexflow_tensor_t,
                                             flexflow_tensor_t,
                                             flexflow_tensor_t *out,
                                             char *name = NULL);
flexflow_error_t
    flexflow_computation_graph_add_op_max(flexflow_computation_graph_t,
                                          flexflow_tensor_t,
                                          flexflow_tensor_t,
                                          flexflow_tensor_t *out,
                                          char *name = NULL);
flexflow_error_t
    flexflow_computation_graph_add_op_min(flexflow_computation_graph_t,
                                          flexflow_tensor_t,
                                          flexflow_tensor_t,
                                          flexflow_tensor_t *out,
                                          char *name = NULL);
flexflow_error_t
    flexflow_computation_graph_add_op_rsqrt(flexflow_computation_graph_t,
                                            flexflow_tensor_t,
                                            flexflow_tensor_t *out,
                                            char *name = NULL);
flexflow_error_t
    flexflow_computation_graph_add_op_pow(flexflow_computation_graph_t,
                                          flexflow_tensor_t,
                                          flexflow_tensor_t *out,
                                          float exponent,
                                          char *name = NULL);
flexflow_error_t flexflow_computation_graph_add_op_scalar_multiply(
    flexflow_computation_graph_t,
    flexflow_tensor_t,
    flexflow_tensor_t *out,
    float scalar,
    char *name = NULL);
flexflow_error_t
    flexflow_computation_graph_add_op_scalar_add(flexflow_computation_graph_t,
                                                 float scalar,
                                                 flexflow_tensor_t,
                                                 flexflow_tensor_t *out,
                                                 char *name = NULL);
flexflow_error_t
    flexflow_computation_graph_add_op_scalar_sub(flexflow_computation_graph_t,
                                                 float scalar,
                                                 flexflow_tensor_t,
                                                 flexflow_tensor_t *out,
                                                 char *name = NULL);
flexflow_error_t flexflow_computation_graph_add_op_scalar_truediv(
    flexflow_computation_graph_t,
    float scalar,
    flexflow_tensor_t,
    flexflow_tensor_t *out,
    char *name = NULL);
flexflow_error_t
    flexflow_computation_graph_add_op_sin(flexflow_computation_graph_t,
                                          flexflow_tensor_t,
                                          flexflow_tensor_t *out,
                                          char *name = NULL);
flexflow_error_t
    flexflow_computation_graph_add_op_cos(flexflow_computation_graph_t,
                                          flexflow_tensor_t,
                                          flexflow_tensor_t *out,
                                          char *name = NULL);
flexflow_error_t
    flexflow_computation_graph_add_op_relu(flexflow_computation_graph_t,
                                           flexflow_tensor_t,
                                           flexflow_tensor_t *out,
                                           char *name = NULL);
flexflow_error_t
    flexflow_computation_graph_add_op_identity(flexflow_computation_graph_t,
                                               flexflow_tensor_t,
                                               flexflow_tensor_t *out,
                                               char *name = NULL);
flexflow_error_t
    flexflow_computation_graph_add_op_gelu(flexflow_computation_graph_t,
                                           flexflow_tensor_t,
                                           flexflow_tensor_t *out,
                                           char *name = NULL);
flexflow_error_t
    flexflow_computation_graph_add_op_sigmoid(flexflow_computation_graph_t,
                                              flexflow_tensor_t,
                                              flexflow_tensor_t *out,
                                              char *name = NULL);
flexflow_error_t
    flexflow_computation_graph_add_op_tanh(flexflow_computation_graph_t,
                                           flexflow_tensor_t,
                                           flexflow_tensor_t *out,
                                           char *name = NULL);
flexflow_error_t
    flexflow_computation_graph_add_op_elu(flexflow_computation_graph_t,
                                          flexflow_tensor_t,
                                          flexflow_tensor_t *out,
                                          char *name = NULL);
flexflow_error_t flexflow_computation_graph_add_op_conv2d(
    flexflow_computation_graph_t,
    flexflow_tensor_t,
    flexflow_tensor_t *out,
    int outChannels,
    int kernelH,
    int kernelW,
    int strideH,
    int strideW,
    int paddingH,
    int paddingW,
    flexflow_activation_t activation,
    int groups = 1,
    bool use_bias = true,
    flexflow_initializer_t kernel_initializer = NO_INITIALIZER,
    flexflow_initializer_t bias_initializer = NO_INITIALIZER,
    char *name = NULL);
flexflow_error_t
    flexflow_computation_graph_add_op_dropout(flexflow_computation_graph_t,
                                              flexflow_tensor_t,
                                              float rate,
                                              unsigned long long seed,
                                              char *name = NULL);
flexflow_error_t flexflow_computation_graph_add_op_embedding(
    flexflow_computation_graph_t,
    flexflow_tensor_t,
    int num_entries,
    int out_dim,
    flexflow_aggregate_op_t aggr_op,
    flexflow_datatype_t output_type,
    flexflow_initializer_t initializer = NO_INITIALIZER,
    char *name = NULL);
flexflow_error_t
    flexflow_computation_graph_add_op_gather(flexflow_computation_graph_t,
                                             flexflow_tensor_t,
                                             flexflow_tensor_t *outs,
                                             int dim,
                                             char *name = NULL);
flexflow_error_t
    flexflow_computation_graph_add_op_group_by(flexflow_computation_graph_t,
                                               flexflow_tensor_t data,
                                               flexflow_tensor_t assign,
                                               flexflow_tensor_t *outs,
                                               int n,
                                               float alpha,
                                               char *name = NULL);
flexflow_error_t flexflow_computation_graph_add_op_cache(
    flexflow_computation_graph_t,
    flexflow_tensor_t,
    flexflow_tensor_t *out,
    int num_batches,
    float (*score_func)(float *, void *, void *, int),
    flexflow_tensor_t assign,
    flexflow_tensor_t *outs,
    int n,
    float alpha,
    char *name = NULL);
flexflow_error_t flexflow_computation_graph_add_op_aggregate(
    flexflow_computation_graph_t,
    flexflow_tensor_t gate_preds,
    flexflow_tensor_t gate_assign,
    flexflow_tensor_t true_gate_assign,
    flexflow_tensor_t full_gate_gradients,
    flexflow_tensor_t *exp_preds,
    flexflow_tensor_t *out,
    int n,
    float lambda_bal,
    char *name = NULL);
flexflow_error_t flexflow_computation_graph_add_op_aggregate_spec(
    flexflow_computation_graph_t,
    flexflow_tensor_t *inputs,
    flexflow_tensor_t *out,
    int n,
    float lambda_bal,
    char *name = NULL);
flexflow_error_t flexflow_computation_graph_add_op_pool2d(
    flexflow_computation_graph_t,
    flexflow_tensor_t,
    flexflow_tensor_t *out,
    int kernelH,
    int kernelW,
    int strideH,
    int strideW,
    int paddingH,
    int paddingW,
    flexflow_pool_op_t pool_op = FLEXFLOW_POOL_OP_MAX,
    flexflow_activation_t activation = FLEXFLOW_ACTIVATION_NONE,
    char *name = NULL);
flexflow_error_t
    flexflow_computation_graph_add_op_layer_norm(flexflow_computation_graph_t,
                                                 flexflow_tensor_t,
                                                 flexflow_tensor_t *out,
                                                 int *axes,
                                                 bool elementwise_affine,
                                                 float eps,
                                                 char *name = NULL);
flexflow_error_t
    flexflow_computation_graph_add_op_batch_norm(flexflow_computation_graph_t,
                                                 flexflow_tensor_t,
                                                 flexflow_tensor_t *out,
                                                 bool relu = true,
                                                 char *name = NULL);
flexflow_error_t
    flexflow_computation_graph_add_op_batch_matmul(flexflow_computation_graph_t,
                                                   flexflow_tensor_t a,
                                                   flexflow_tensor_t b,
                                                   flexflow_tensor_t *out,
                                                   int a_seq_length_dim = -1,
                                                   int b_seq_length_dim = -1,
                                                   char *name = NULL);
flexflow_error_t flexflow_computation_graph_add_op_dense(
    flexflow_computation_graph_t,
    flexflow_tensor_t,
    flexflow_tensor_t *out,
    int out_dim,
    flexflow_activation_t activation = FLEXFLOW_ACTIVATION_NONE,
    bool use_bias = true,
    flexflow_datatype_t compute_type = FLEXFLOW_DATATYPE_FLOAT,
    flexflow_initializer_t kernel_initializer = NO_INITIALIZER,
    flexflow_initializer_t bias_initializer = NO_INITIALIZER,
    char *name = NULL);
flexflow_error_t
    flexflow_computation_graph_add_op_cast(flexflow_computation_graph_t,
                                           flexflow_tensor_t,
                                           flexflow_tensor_t *out,
                                           flexflow_datatype_t out_type,
                                           char *name = NULL);
flexflow_error_t
    flexflow_computation_graph_add_op_concat(flexflow_computation_graph_t,
                                             flexflow_tensor_t *inputs,
                                             flexflow_tensor_t *out,
                                             int num_inputs,
                                             int axis,
                                             char *name = NULL);
flexflow_error_t
    flexflow_computation_graph_add_op_mean(flexflow_computation_graph_t,
                                           flexflow_tensor_t,
                                           flexflow_tensor_t *out,
                                           int *dims,
                                           bool keepdims,
                                           char *name = NULL);
flexflow_error_t
    flexflow_computation_graph_add_op_moe(flexflow_computation_graph_t,
                                          flexflow_tensor_t,
                                          flexflow_tensor_t *out,
                                          int num_exp,
                                          int num_select,
                                          int expert_hidden_size,
                                          float alpha,
                                          float lambda,
                                          char *name = NULL);
flexflow_error_t
    flexflow_computation_graph_add_op_split(flexflow_computation_graph_t,
                                            flexflow_tensor_t,
                                            flexflow_tensor_t *outs,
                                            int *splits,
                                            int axis,
                                            char *name = NULL);
flexflow_error_t
    flexflow_computation_graph_add_op_flat(flexflow_computation_graph_t,
                                           flexflow_tensor_t,
                                           flexflow_tensor_t *out,
                                           char *name = NULL);
flexflow_error_t
    flexflow_computation_graph_add_op_softmax(flexflow_computation_graph_t,
                                              flexflow_tensor_t,
                                              flexflow_tensor_t *out,
                                              int dim = -1,
                                              char *name = NULL);
flexflow_error_t
    flexflow_computation_graph_add_op_transpose(flexflow_computation_graph_t,
                                                flexflow_tensor_t,
                                                int *permutation,
                                                char *name = NULL);
flexflow_error_t
    flexflow_computation_graph_add_op_reduce_sum(flexflow_computation_graph_t,
                                                 flexflow_tensor_t,
                                                 flexflow_tensor_t *out,
                                                 int *axes,
                                                 bool keepdims = false,
                                                 char *name = NULL);
flexflow_error_t
    flexflow_computation_graph_add_op_reshape(flexflow_computation_graph_t,
                                              flexflow_tensor_t,
                                              flexflow_tensor_t *out,
                                              int *shape,
                                              char *name = NULL);
flexflow_error_t
    flexflow_computation_graph_add_op_reverse(flexflow_computation_graph_t,
                                              flexflow_tensor_t,
                                              flexflow_tensor_t *out,
                                              int axis,
                                              char *name = NULL);
flexflow_error_t
    flexflow_computation_graph_add_op_topk(flexflow_computation_graph_t,
                                           flexflow_tensor_t,
                                           flexflow_tensor_t *outs,
                                           int k,
                                           bool sorted,
                                           char *name = NULL);
flexflow_error_t flexflow_computation_graph_add_multihead_attention(
    flexflow_computation_graph_t,
    flexflow_tensor_t query,
    flexflow_tensor_t key,
    flexflow_tensor_t value,
    int embed_dim,
    int num_heads,
    int kdim = 0,
    int vdim = 0,
    float dropout = 0.0f,
    bool bias = true,
    bool add_bias_kv = false,
    flexflow_initializer_t initializer = NO_INITIALIZER,
    char *name = NULL);

FLEXFLOW_FFI_END();

#endif
