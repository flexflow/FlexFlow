#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_COMPUTATION_GRAPH_BUILDER_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_COMPUTATION_GRAPH_BUILDER_H

#include "computation_graph.h"

namespace FlexFlow {

tensor_guid_t insert_exp_layer(ComputationGraph &,
                               tensor_guid_t const &,
                               optional<std::string> const &name = nullopt);

tensor_guid_t insert_add_layer(ComputationGraph &,
                               tensor_guid_t const &x,
                               tensor_guid_t const &y,
                               optional<std::string> const &name = nullopt);

tensor_guid_t
    insert_subtract_layer(ComputationGraph &,
                          tensor_guid_t const &x,
                          tensor_guid_t const &y,
                          optional<std::string> const &name = nullopt);

tensor_guid_t
    insert_multiply_layer(ComputationGraph &,
                          tensor_guid_t const &x,
                          tensor_guid_t const &y,
                          optional<std::string> const &name = nullopt);

tensor_guid_t insert_divide_layer(ComputationGraph &,
                                  tensor_guid_t const &x,
                                  tensor_guid_t const &y,
                                  optional<std::string> const &name = nullopt);

tensor_guid_t insert_max_layer(ComputationGraph &,
                               tensor_guid_t const &x,
                               tensor_guid_t const &y,
                               optional<std::string> const &name = nullopt);

tensor_guid_t insert_min_layer(ComputationGraph &,
                               tensor_guid_t const &x,
                               tensor_guid_t const &y,
                               optional<std::string> const &name = nullopt);

tensor_guid_t insert_rsqrt_layer(ComputationGraph &,
                                 tensor_guid_t const &x,
                                 optional<std::string> const &name = nullopt);

tensor_guid_t insert_pow_layer(ComputationGraph &,
                               tensor_guid_t const &x,
                               float exponent,
                               optional<std::string> const &name = nullopt);

tensor_guid_t
    insert_scalar_multiply_layer(ComputationGraph &,
                                 tensor_guid_t const &x,
                                 float scalar,
                                 optional<std::string> const &name = nullopt);

tensor_guid_t
    insert_scalar_add_layer(ComputationGraph &,
                            tensor_guid_t const &x,
                            float scalar,
                            optional<std::string> const &name = nullopt);

tensor_guid_t
    insert_scalar_sub_layer(ComputationGraph &,
                            tensor_guid_t const &lhs,
                            float rhs,
                            optional<std::string> const &name = nullopt);

tensor_guid_t
    insert_scalar_truediv_layer(ComputationGraph &,
                                tensor_guid_t const &numerator,
                                float denominator,
                                optional<std::string> const &name = nullopt);

tensor_guid_t insert_sin_layer(ComputationGraph &,
                               tensor_guid_t const &x,
                               optional<std::string> const &name = nullopt);

tensor_guid_t insert_cos_layer(ComputationGraph &,
                               tensor_guid_t const &x,
                               optional<std::string> const &name = nullopt);

tensor_guid_t insert_relu_layer(ComputationGraph &,
                                tensor_guid_t const &x,
                                optional<std::string> const &name = nullopt);
tensor_guid_t
    insert_identity_layer(ComputationGraph &,
                          tensor_guid_t const &x,
                          optional<std::string> const &name = nullopt);
tensor_guid_t insert_gelu_layer(ComputationGraph &,
                                tensor_guid_t const &x,
                                optional<std::string> const &name = nullopt);
tensor_guid_t insert_sigmoid_layer(ComputationGraph &,
                                   tensor_guid_t const &x,
                                   optional<std::string> const &name = nullopt);
tensor_guid_t insert_tanh_layer(ComputationGraph &,
                                tensor_guid_t const &x,
                                optional<std::string> const &name = nullopt);
tensor_guid_t insert_elu_layer(ComputationGraph &,
                               tensor_guid_t const &x,
                               optional<std::string> const &name = nullopt);

tensor_guid_t insert_conv2d_layer(
    ComputationGraph &,
    tensor_guid_t const &input,
    int outChannels,
    int kernelH,
    int kernelW,
    int strideH,
    int strideW,
    int paddingH,
    int paddingW,
    optional<Activation> const &activation = nullopt,
    int groups = 1,
    bool use_bias = true,
    optional<Initializer> const &kernel_initializer = nullopt,
    optional<Initializer> const &bias_initializer = nullopt,
    optional<RegularizerAttrs> const &kernel_regularizer = nullopt,
    optional<std::string> const &name = nullopt);

tensor_guid_t insert_dropout_layer(ComputationGraph &,
                                   tensor_guid_t const &input,
                                   float rate,
                                   unsigned long long seed = 0,
                                   optional<std::string> const &name = nullopt);

tensor_guid_t insert_embedding_layer(
    ComputationGraph &,
    tensor_guid_t const &input,
    int num_entries,
    int outDim,
    AggregateOp aggr,
    DataType dtype = DataType::FLOAT,
    optional<Initializer> const &kernel_initializer = nullopt,
    optional<std::string> const &name = nullopt);

std::vector<tensor_guid_t>
    insert_gather_layer(ComputationGraph &,
                        tensor_guid_t const &input,
                        tensor_guid_t const &index,
                        ff_dim_t dim,
                        optional<std::string> const &name = nullopt);

std::vector<tensor_guid_t>
    insert_group_by_layer(ComputationGraph &,
                          tensor_guid_t const &data,
                          tensor_guid_t const &assign,
                          int n,
                          float alpha,
                          optional<std::string> const &name = nullopt);

tensor_guid_t insert_cache_layer(
    ComputationGraph &,
    tensor_guid_t const &input,
    int num_batches,
    std::function<float(float *, void const *, void const *, int)> score_f = {},
    optional<std::string> const &name = nullopt);

tensor_guid_t
    insert_aggregate_layer(ComputationGraph &,
                           tensor_guid_t const &gate_preds,
                           tensor_guid_t const &gate_assign,
                           tensor_guid_t const &true_gate_assign,
                           tensor_guid_t const &full_gate_gradients,
                           std::vector<tensor_guid_t> const &exp_preds,
                           float lambda_bal,
                           optional<std::string> const &name = nullopt);

tensor_guid_t
    insert_aggregate_spec_layer(ComputationGraph &,
                                std::vector<tensor_guid_t> const &inputs,
                                float lambda_bal,
                                optional<std::string> const &name = nullopt);

tensor_guid_t
    insert_pool2d_layer(ComputationGraph &,
                        tensor_guid_t const &input,
                        int kernelH,
                        int kernelW,
                        int strideH,
                        int strideW,
                        int paddingH,
                        int paddingW,
                        PoolOp type = PoolOp::MAX,
                        optional<Activation> const &activation = nullopt,
                        optional<std::string> const &name = nullopt);

tensor_guid_t
    insert_layer_norm_layer(ComputationGraph &,
                            tensor_guid_t const &input,
                            std::vector<int> const &axes,
                            bool elementwise_affine,
                            float eps,
                            optional<std::string> const &name = nullopt);

tensor_guid_t
    insert_batch_norm_layer(ComputationGraph &,
                            tensor_guid_t const &input,
                            bool relu = true,
                            optional<std::string> const &name = nullopt);
tensor_guid_t
    insert_batch_matmul_layer(ComputationGraph &,
                              tensor_guid_t const &A,
                              tensor_guid_t const &B,
                              int a_seq_length_dim = -1,
                              int b_seq_length_dim = -1,
                              optional<std::string> const &name = nullopt);
tensor_guid_t insert_dense_layer(
    ComputationGraph &,
    tensor_guid_t const &input,
    int outDim,
    optional<Activation> activation = nullopt,
    bool use_bias = true,
    DataType data_type = DataType::FLOAT,
    optional<Initializer> const &kernel_initializer = nullopt,
    optional<Initializer> const &bias_initializer = nullopt,
    optional<std::string> const &name = nullopt);

tensor_guid_t insert_cast_layer(ComputationGraph &,
                                tensor_guid_t const &input,
                                DataType dtype,
                                optional<std::string> const &name = nullopt);

tensor_guid_t insert_concat_layer(ComputationGraph &,
                                  std::vector<tensor_guid_t> const &tensors,
                                  int axis,
                                  optional<std::string> const &name = nullopt);

tensor_guid_t insert_mean_layer(ComputationGraph &,
                                tensor_guid_t const &input,
                                std::vector<int> const &dims,
                                bool keepdims,
                                optional<std::string> const &name = nullopt);

tensor_guid_t
    insert_moe_layer_composite(ComputationGraph &,
                               tensor_guid_t const &input,
                               int num_exp,
                               int num_select,
                               int expert_hidden_size,
                               float alpha,
                               float lambda,
                               optional<std::string> const &name = nullopt);

std::vector<tensor_guid_t>
    insert_split_layer(ComputationGraph &,
                       tensor_guid_t const &input,
                       std::vector<int> const &split,
                       int axis,
                       optional<std::string> const &name = nullopt);

tensor_guid_t insert_flat_layer(ComputationGraph &,
                                tensor_guid_t const &input,
                                optional<std::string> const &name = nullopt);

tensor_guid_t insert_softmax_layer(ComputationGraph &,
                                   tensor_guid_t const &input,
                                   int dim = -1,
                                   optional<std::string> const &name = nullopt);

tensor_guid_t
    insert_transpose_layer(ComputationGraph &,
                           tensor_guid_t const &input,
                           std::vector<int> const &perm,
                           optional<std::string> const &name = nullopt);

tensor_guid_t
    insert_reduce_sum_layer(ComputationGraph &,
                            tensor_guid_t const &input,
                            std::vector<int> const &axes,
                            bool keepdims = false,
                            optional<std::string> const &name = nullopt);

tensor_guid_t insert_reshape_layer(ComputationGraph &,
                                   tensor_guid_t const &input,
                                   std::vector<int> const &shape,
                                   optional<std::string> const &name = nullopt);

tensor_guid_t insert_reverse_layer(ComputationGraph &,
                                   tensor_guid_t const &input,
                                   int axis,
                                   optional<std::string> const &name = nullopt);

std::vector<tensor_guid_t>
    insert_top_k_layer(ComputationGraph &,
                       tensor_guid_t const &input,
                       int k,
                       bool sorted,
                       optional<std::string> const &name = nullopt);

tensor_guid_t insert_multihead_attention_layer(
    ComputationGraph &,
    tensor_guid_t const &query,
    tensor_guid_t const &key,
    tensor_guid_t const &value,
    int embed_dim,
    int num_heads,
    int kdim = 0,
    int vdim = 0,
    float dropout = 0.0f,
    bool bias = true,
    bool add_bias_kv = false,
    bool add_zero_attn = false,
    optional<Initializer> const &initializer = nullopt,
    optional<std::string> const &name = nullopt);

tensor_guid_t insert_new_activation_tensor(ComputationGraph &,
                                           TensorShape const &);
weight_guid_t insert_new_weight_tensor(
    ComputationGraph &,
    TensorShape const &,
    optional<Initializer> const &initializer = nullopt);

tensor_guid_t insert_broadcast_layer(tensor_guid_t const &,
                                     TensorShape const &);

} // namespace FlexFlow

#endif
