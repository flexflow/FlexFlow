#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_COMPUTATION_GRAPH_BUILDER_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_COMPUTATION_GRAPH_BUILDER_H

#include "computation_graph.h"

namespace FlexFlow {

struct ComputationGraphBuilder
    : public use_visitable_cmp<ComputationGraphBuilder> {
public:
  ComputationGraphBuilder();

  // C++ APIs for constructing models
  // Add an exp layer
  Tensor exp(Tensor const &, optional<std::string> const &name = nullopt);
  // Add an add layer
  Tensor add(Tensor const &x,
             Tensor const &y,
             optional<std::string> const &name = nullopt);
  // Add a subtract layer
  Tensor subtract(Tensor const &x,
                  Tensor const &y,
                  optional<std::string> const &name = nullopt);
  // Add a multiply layer
  Tensor multiply(Tensor const &x,
                  Tensor const &y,
                  optional<std::string> const &name = nullopt);
  // Add a divide layer
  Tensor divide(Tensor const &x,
                Tensor const &y,
                optional<std::string> const &name = nullopt);
  // Add a max layer
  Tensor max(Tensor const &x,
             Tensor const &y,
             optional<std::string> const &name = nullopt);
  // Add a min layer
  Tensor min(Tensor const &x,
             Tensor const &y,
             optional<std::string> const &name = nullopt);
  // Add a rsqrt layer
  Tensor rsqrt(Tensor const &x, optional<std::string> const &name = nullopt);
  // Add a pow layer
  Tensor pow(Tensor const &x,
             float exponent,
             optional<std::string> const &name = nullopt);
  // Add a scalar multiply layer
  Tensor scalar_multiply(Tensor const &x,
                         float scalar,
                         optional<std::string> const &name = nullopt);
  Tensor scalar_add(Tensor const &x,
                    float scalar,
                    optional<std::string> const &name = nullopt);
  Tensor scalar_sub(Tensor const &lhs,
                    float rhs,
                    optional<std::string> const &name = nullopt);
  Tensor scalar_truediv(Tensor const &numerator,
                        float denominator,
                        optional<std::string> const &name = nullopt);
  // Add a sin layer
  Tensor sin(Tensor const &x, optional<std::string> const &name = nullopt);
  // Add a cos layer
  Tensor cos(Tensor const &x, optional<std::string> const &name = nullopt);
  // Add an activation layer
  Tensor relu(Tensor const &x, optional<std::string> const &name = nullopt);
  Tensor identity(Tensor const &x, optional<std::string> const &name = nullopt);
  Tensor gelu(Tensor const &x, optional<std::string> const &name = nullopt);
  Tensor sigmoid(Tensor const &x, optional<std::string> const &name = nullopt);
  Tensor tanh(Tensor const &x, optional<std::string> const &name = nullopt);
  Tensor elu(Tensor const &x, optional<std::string> const &name = nullopt);
  // Add a 2D convolutional layer
  Tensor conv2d(Tensor const &input,
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
                optional<Initializer const &> kernel_initializer = nullopt,
                optional<Initializer const &> bias_initializer = nullopt,
                optional<RegularizerAttrs const &> kernel_regularizer = nullopt,
                optional<std::string> const &name = nullopt);
  // Add a dropout layer
  Tensor dropout(Tensor const &input,
                 float rate,
                 unsigned long long seed = 0,
                 optional<std::string> const &name = nullopt);
  // Add an embedding layer
  Tensor embedding(Tensor const &input,
                   int num_entries,
                   int outDim,
                   AggregateOp aggr,
                   DataType dtype = DataType::FLOAT,
                   optional<Initializer const &> kernel_initializer = nullopt,
                   optional<std::string> const &name = nullopt);
  // Add a gather layer
  std::vector<Tensor> gather(Tensor const &input,
                             Tensor const &index,
                             ff_dim_t dim,
                             optional<std::string> const &name = nullopt);
  // Add a group_by layer
  void group_by(Tensor const &data,
                Tensor const &assign,
                Tensor *outputs,
                int n,
                float alpha,
                optional<std::string> const &name = nullopt);
  // Add a cache layer
  Tensor cache(Tensor const &input,
               int num_batches,
               std::function<float(float *, void const *, void const *, int)>
                   score_f = {},
               optional<std::string> const &name = nullopt);
  // Add aggregate layer
  Tensor aggregate(Tensor const &gate_preds,
                   Tensor const &gate_assign,
                   Tensor const &true_gate_assign,
                   Tensor const &full_gate_gradients,
                   std::vector<Tensor> const &exp_preds,
                   int n,
                   float lambda_bal,
                   optional<std::string> const &maybe_name);
  // Add aggregate_spec layer
  Tensor aggregate_spec(std::vector<Tensor> const &inputs,
                        int n,
                        float lambda_bal,
                        optional<std::string> const &name = nullopt);
  // Add a 2D pooling layer
  Tensor pool2d(Tensor const &input,
                int kernelH,
                int kernelW,
                int strideH,
                int strideW,
                int paddingH,
                int paddingW,
                PoolOp type = PoolOp::MAX,
                optional<Activation> const &activation = nullopt,
                optional<std::string> const &name = nullopt);
  Tensor layer_norm(Tensor const &input,
                    std::vector<int> const &axes,
                    bool elementwise_affine,
                    float eps,
                    optional<std::string> const &name = nullopt);
  Tensor batch_norm(Tensor const &input,
                    bool relu = true,
                    optional<std::string> const &name = nullopt);
  Tensor batch_matmul(Tensor const &A,
                      Tensor const &B,
                      int a_seq_length_dim = -1,
                      int b_seq_length_dim = -1,
                      optional<std::string> const &name = nullopt);
  Tensor dense(Tensor const &input,
               int outDim,
               optional<Activation> activation = nullopt,
               bool use_bias = true,
               DataType data_type = DataType::FLOAT,
               optional<Initializer const &> kernel_initializer = nullopt,
               optional<Initializer const &> bias_initializer = nullopt,
               optional<std::string> const &name = nullopt);
  // Add a cast layer
  Tensor cast(Tensor const &input,
              DataType dtype,
              optional<std::string> const &name = nullopt);
  // Add a concat layer
  Tensor concat(int n,
                std::vector<Tensor> const &tensors,
                int axis,
                optional<std::string> const &name = nullopt);
  // Add a mean layer
  Tensor mean(Tensor const &input,
              std::vector<int> const &dims,
              bool keepdims,
              char const *name);
  // Add a moe layer (wrapping topk, group_by and aggregate operators)
  Tensor moe(Tensor const &input,
             int num_exp,
             int num_select,
             int expert_hidden_size,
             float alpha,
             float lambda);
  // Add a split layer
  void split(Tensor const &input,
             Tensor *outputs,
             std::vector<int> const &split,
             int axis,
             optional<std::string> const &name = nullopt);
  // Add a flat layer
  Tensor flat(Tensor const &input, optional<std::string> const &name = nullopt);
  // Add a softmax layer
  Tensor softmax(Tensor const &input,
                 int dim = -1,
                 optional<std::string> const &name = nullopt);
  // Create input tensors and constants
  Tensor transpose(Tensor const &input,
                   std::vector<int> const &perm,
                   optional<std::string> const &name = nullopt);
  Tensor reduce_sum(Tensor const &input,
                    std::vector<int> const &axes,
                    bool keepdims = false,
                    optional<std::string> const &name = nullopt);
  Tensor reshape(Tensor const &input,
                 std::vector<int> const &shape,
                 optional<std::string> const &name = nullopt);
  Tensor reverse(Tensor const &input,
                 int axis,
                 optional<std::string> const &name = nullopt);
  void top_k(Tensor const &input,
             Tensor *outputs,
             int k,
             bool sorted,
             optional<std::string> const &name = nullopt);
  Tensor
      multihead_attention(Tensor const &query,
                          Tensor const &key,
                          Tensor const &value,
                          int embed_dim,
                          int num_heads,
                          int kdim = 0,
                          int vdim = 0,
                          float dropout = 0.0f,
                          bool bias = true,
                          bool add_bias_kv = false,
                          bool add_zero_attn = false,
                          optional<Initializer const &> initializer = nullopt,
                          optional<std::string> const &name = nullopt);
  Tensor create_tensor(TensorShape const &, bool create_grad = true);
  Parameter create_weight(TensorShape const &,
                          bool create_grad = true,
                          optional<Initializer const &> initializer = nullopt,
                          optional<ParamSync> sync_type = nullopt);

  std::vector<Tensor> get_outputs(Layer const &) const;
  Tensor get_output(Layer const &, int idx) const;

  Tensor at(MultiDiEdge const &) const;
  Layer at(Node const &) const;

private:
  Tensor broadcast(Tensor const &, TensorShape const &);

  void add_layer(Layer const &layer,
                 std::vector<Tensor> const &inputs,
                 std::vector<Tensor> const &weights,
                 std::vector<Tensor> const &outputs);
  Tensor
      add_layer(Layer const &layer,
                std::vector<Tensor> const &inputs,
                std::vector<std::pair<TensorShape, optional<Initializer>>> const
                    &weight_shapes,
                TensorShape const &output_shape);
  std::vector<Tensor>
      add_layer(Layer const &layer,
                std::vector<Tensor> const &inputs,
                std::vector<std::pair<TensorShape, optional<Initializer>>> const
                    &weight_shapes,
                std::vector<TensorShape> const &output_shapes);

  Tensor as_type(Tensor const &, DataType, std::string const &);

  TensorShape get_broadcast_target_shape(std::vector<TensorShape> const &);

  Tensor element_binary(OperatorType,
                        Tensor const &lhs,
                        Tensor const &rhs,
                        optional<std::string> const &name = nullopt);

  Tensor element_unary(OperatorType,
                       Tensor const &input,
                       optional<std::string> const &name = nullopt);
  Tensor element_scalar_unary(OperatorType,
                              Tensor const &input,
                              float scalar,
                              optional<std::string> const &name = nullopt);
  Tensor
      element_unary(variant<ElementUnaryAttrs, ElementScalarUnaryAttrs> const &,
                    Tensor const &input,
                    optional<std::string> const &name = nullopt);

public:
  ComputationGraph computation_graph;
};

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::ComputationGraphBuilder, computation_graph);

namespace FlexFlow {
static_assert(
    is_well_behaved_value_type_no_hash<ComputationGraphBuilder>::value, "");
}

#endif
