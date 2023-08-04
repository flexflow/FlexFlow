#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_COMPUTATION_GRAPH_BUILDER_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_COMPUTATION_GRAPH_BUILDER_H

#include "computation_graph.h"

namespace FlexFlow {
//
// C++ APIs for constructing models
// Add an exp layer
Tensor exp(ComputationGraph &, Tensor const &, optional<std::string> const &name = nullopt);
// Add an add layer
Tensor add(ComputationGraph &,
           Tensor const &x,
           Tensor const &y,
           optional<std::string> const &name = nullopt);
// Add a subtract layer
Tensor subtract(ComputationGraph &,
                Tensor const &x,
                Tensor const &y,
                optional<std::string> const &name = nullopt);
// Add a multiply layer
Tensor multiply(ComputationGraph &,
                Tensor const &x,
                Tensor const &y,
                optional<std::string> const &name = nullopt);
// Add a divide layer
Tensor divide(ComputationGraph &,
              Tensor const &x,
              Tensor const &y,
              optional<std::string> const &name = nullopt);
// Add a max layer
Tensor max(ComputationGraph &,
           Tensor const &x,
           Tensor const &y,
           optional<std::string> const &name = nullopt);
// Add a min layer
Tensor min(ComputationGraph &,
           Tensor const &x,
           Tensor const &y,
           optional<std::string> const &name = nullopt);
// Add a rsqrt layer
Tensor rsqrt(ComputationGraph &, Tensor const &x, optional<std::string> const &name = nullopt);
// Add a pow layer
Tensor pow(ComputationGraph &,
           Tensor const &x,
           float exponent,
           optional<std::string> const &name = nullopt);
// Add a scalar multiply layer
Tensor scalar_multiply(ComputationGraph &, 
                       Tensor const &x,
                       float scalar,
                       optional<std::string> const &name = nullopt);
Tensor scalar_add(ComputationGraph &,
                  Tensor const &x,
                  float scalar,
                  optional<std::string> const &name = nullopt);
Tensor scalar_sub(ComputationGraph &,
                  Tensor const &lhs,
                  float rhs,
                  optional<std::string> const &name = nullopt);
Tensor scalar_truediv(ComputationGraph &,
                      Tensor const &numerator,
                      float denominator,
                      optional<std::string> const &name = nullopt);
// Add a sin layer
Tensor sin(ComputationGraph &, Tensor const &x, optional<std::string> const &name = nullopt);
// Add a cos layer
Tensor cos(ComputationGraph &, Tensor const &x, optional<std::string> const &name = nullopt);
// Add an activation layer
Tensor relu(ComputationGraph &, Tensor const &x, optional<std::string> const &name = nullopt);
Tensor identity(ComputationGraph &, Tensor const &x, optional<std::string> const &name = nullopt);
Tensor gelu(ComputationGraph &, Tensor const &x, optional<std::string> const &name = nullopt);
Tensor sigmoid(ComputationGraph &, Tensor const &x, optional<std::string> const &name = nullopt);
Tensor tanh(ComputationGraph &, Tensor const &x, optional<std::string> const &name = nullopt);
Tensor elu(ComputationGraph &, Tensor const &x, optional<std::string> const &name = nullopt);
// Add a 2D convolutional layer
Tensor conv2d(ComputationGraph &, 
              Tensor const &input,
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
// Add a dropout layer
Tensor dropout(ComputationGraph &,
               Tensor const &input,
               float rate,
               unsigned long long seed = 0,
               optional<std::string> const &name = nullopt);
// Add an embedding layer
Tensor embedding(ComputationGraph &, 
                 Tensor const &input,
                 int num_entries,
                 int outDim,
                 AggregateOp aggr,
                 DataType dtype = DataType::FLOAT,
                 optional<Initializer> const &kernel_initializer = nullopt,
                 optional<std::string> const &name = nullopt);
// Add a gather layer
std::vector<Tensor> gather(ComputationGraph &, 
                           Tensor const &input,
                           Tensor const &index,
                           ff_dim_t dim,
                           optional<std::string> const &name = nullopt);
// Add a group_by layer
std::vector<Tensor> group_by(ComputationGraph &,
                             Tensor const &data,
              Tensor const &assign,
              int n,
              float alpha,
              optional<std::string> const &name = nullopt);
// Add a cache layer
Tensor cache(ComputationGraph &, 
             Tensor const &input,
             int num_batches,
             std::function<float(float *, void const *, void const *, int)>
                 score_f = {},
             optional<std::string> const &name = nullopt);
// Add aggregate layer
Tensor aggregate(ComputationGraph &,
                 Tensor const &gate_preds,
                 Tensor const &gate_assign,
                 Tensor const &true_gate_assign,
                 Tensor const &full_gate_gradients,
                 std::vector<Tensor> const &exp_preds,
                 float lambda_bal,
                 optional<std::string> const &name = nullopt);
// Add aggregate_spec layer
Tensor aggregate_spec(ComputationGraph &,
                      std::vector<Tensor> const &inputs,
                      float lambda_bal,
                      optional<std::string> const &name = nullopt);
// Add a 2D pooling layer
Tensor pool2d(ComputationGraph &,
              Tensor const &input,
              int kernelH,
              int kernelW,
              int strideH,
              int strideW,
              int paddingH,
              int paddingW,
              PoolOp type = PoolOp::MAX,
              optional<Activation> const &activation = nullopt,
              optional<std::string> const &name = nullopt);
Tensor layer_norm(ComputationGraph &,
                  Tensor const &input,
                  std::vector<int> const &axes,
                  bool elementwise_affine,
                  float eps,
                  optional<std::string> const &name = nullopt);
Tensor batch_norm(ComputationGraph &, 
                  Tensor const &input,
                  bool relu = true,
                  optional<std::string> const &name = nullopt);
Tensor batch_matmul(ComputationGraph &, 
                    Tensor const &A,
                    Tensor const &B,
                    int a_seq_length_dim = -1,
                    int b_seq_length_dim = -1,
                    optional<std::string> const &name = nullopt);
Tensor dense(ComputationGraph &,
             Tensor const &input,
             int outDim,
             optional<Activation> activation = nullopt,
             bool use_bias = true,
             DataType data_type = DataType::FLOAT,
             optional<Initializer> const &kernel_initializer = nullopt,
             optional<Initializer> const &bias_initializer = nullopt,
             optional<std::string> const &name = nullopt);
// Add a cast layer
Tensor cast(ComputationGraph &, 
            Tensor const &input,
            DataType dtype,
            optional<std::string> const &name = nullopt);
// Add a concat layer
Tensor concat(ComputationGraph &,
              std::vector<Tensor> const &tensors,
              int axis,
              optional<std::string> const &name = nullopt);
// Add a mean layer
Tensor mean(ComputationGraph &, 
            Tensor const &input,
            std::vector<int> const &dims,
            bool keepdims,
            optional<std::string> const &name = nullopt);
// Add a moe layer (wrapping topk, group_by and aggregate operators)
Tensor moe(ComputationGraph &, 
           Tensor const &input,
           int num_exp,
           int num_select,
           int expert_hidden_size,
           float alpha,
           float lambda,
           optional<std::string> const &name = nullopt);
// Add a split layer
std::vector<Tensor> split(ComputationGraph &, 
                          Tensor const &input,
           std::vector<int> const &split,
           int axis,
           optional<std::string> const &name = nullopt);
// Add a flat layer
Tensor flat(ComputationGraph &, Tensor const &input, optional<std::string> const &name = nullopt);
// Add a softmax layer
Tensor softmax(ComputationGraph &,
               Tensor const &input,
               int dim = -1,
               optional<std::string> const &name = nullopt);
// Create input tensors and constants
Tensor transpose(ComputationGraph &,
                 Tensor const &input,
                 std::vector<int> const &perm,
                 optional<std::string> const &name = nullopt);
Tensor reduce_sum(ComputationGraph &,
                  Tensor const &input,
                  std::vector<int> const &axes,
                  bool keepdims = false,
                  optional<std::string> const &name = nullopt);
Tensor reshape(ComputationGraph &,
               Tensor const &input,
               std::vector<int> const &shape,
               optional<std::string> const &name = nullopt);
Tensor reverse(ComputationGraph &,
               Tensor const &input,
               int axis,
               optional<std::string> const &name = nullopt);
std::vector<Tensor> top_k(ComputationGraph &,
                          Tensor const &input,
           int k,
           bool sorted,
           optional<std::string> const &name = nullopt);
Tensor
    multihead_attention(ComputationGraph &,
                        Tensor const &query,
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
                        optional<Initializer> const &initializer = nullopt,
                        optional<std::string> const &name = nullopt);
Tensor create_tensor(ComputationGraph &, TensorShape const &, bool create_grad = true);
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


} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::ComputationGraphBuilder, computation_graph);

namespace FlexFlow {
static_assert(
    is_well_behaved_value_type_no_hash<ComputationGraphBuilder>::value, "");
}

#endif
