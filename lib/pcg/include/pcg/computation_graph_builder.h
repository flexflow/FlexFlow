#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_COMPUTATION_GRAPH_BUILDER_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_COMPUTATION_GRAPH_BUILDER_H

#include "pcg/computation_graph.dtg.h"
#include "pcg/initializer_attrs.dtg.h"
#include "pcg/tensor_guid_t.dtg.h"

namespace FlexFlow {

struct ComputationGraphBuilder {
public:
  ComputationGraphBuilder();

  // C++ APIs for constructing models
  // Add an exp layer
  tensor_guid_t exp(tensor_guid_t const &,
                    std::optional<std::string> const &name = std::nullopt);
  // Add an add layer
  tensor_guid_t add(tensor_guid_t const &x,
                    tensor_guid_t const &y,
                    std::optional<std::string> const &name = std::nullopt);
  // Add a subtract layer
  tensor_guid_t subtract(tensor_guid_t const &x,
                         tensor_guid_t const &y,
                         std::optional<std::string> const &name = std::nullopt);
  // Add a multiply layer
  tensor_guid_t multiply(tensor_guid_t const &x,
                         tensor_guid_t const &y,
                         std::optional<std::string> const &name = std::nullopt);
  // Add a divide layer
  tensor_guid_t divide(tensor_guid_t const &x,
                       tensor_guid_t const &y,
                       std::optional<std::string> const &name = std::nullopt);
  // Add a max layer
  tensor_guid_t max(tensor_guid_t const &x,
                    tensor_guid_t const &y,
                    std::optional<std::string> const &name = std::nullopt);
  // Add a min layer
  tensor_guid_t min(tensor_guid_t const &x,
                    tensor_guid_t const &y,
                    std::optional<std::string> const &name = std::nullopt);
  // Add a rsqrt layer
  tensor_guid_t rsqrt(tensor_guid_t const &x,
                      std::optional<std::string> const &name = std::nullopt);
  // Add a pow layer
  tensor_guid_t pow(tensor_guid_t const &x,
                    float exponent,
                    std::optional<std::string> const &name = std::nullopt);
  // Add a scalar multiply layer
  tensor_guid_t
      scalar_multiply(tensor_guid_t const &x,
                      float scalar,
                      std::optional<std::string> const &name = std::nullopt);
  tensor_guid_t
      scalar_add(tensor_guid_t const &x,
                 float scalar,
                 std::optional<std::string> const &name = std::nullopt);
  tensor_guid_t
      scalar_sub(tensor_guid_t const &lhs,
                 float rhs,
                 std::optional<std::string> const &name = std::nullopt);
  tensor_guid_t
      scalar_truediv(tensor_guid_t const &numerator,
                     float denominator,
                     std::optional<std::string> const &name = std::nullopt);
  // Add a sin layer
  tensor_guid_t sin(tensor_guid_t const &x,
                    std::optional<std::string> const &name = std::nullopt);
  // Add a cos layer
  tensor_guid_t cos(tensor_guid_t const &x,
                    std::optional<std::string> const &name = std::nullopt);
  // Add an activation layer
  tensor_guid_t relu(tensor_guid_t const &x,
                     std::optional<std::string> const &name = std::nullopt);
  tensor_guid_t identity(tensor_guid_t const &x,
                         std::optional<std::string> const &name = std::nullopt);
  tensor_guid_t gelu(tensor_guid_t const &x,
                     std::optional<std::string> const &name = std::nullopt);
  tensor_guid_t sigmoid(tensor_guid_t const &x,
                        std::optional<std::string> const &name = std::nullopt);
  tensor_guid_t tanh(tensor_guid_t const &x,
                     std::optional<std::string> const &name = std::nullopt);
  tensor_guid_t elu(tensor_guid_t const &x,
                    std::optional<std::string> const &name = std::nullopt);
  // Add a 2D convolutional layer
  tensor_guid_t conv2d(
      tensor_guid_t const &input,
      int outChannels,
      int kernelH,
      int kernelW,
      int strideH,
      int strideW,
      int paddingH,
      int paddingW,
      std::optional<Activation> const &activation = std::nullopt,
      int groups = 1,
      bool use_bias = true,
      std::optional<InitializerAttrs> const &kernel_initializer = std::nullopt,
      std::optional<InitializerAttrs> const &bias_initializer = std::nullopt,
      std::optional<RegularizerAttrs> const &kernel_regularizer = std::nullopt,
      std::optional<std::string> const &name = std::nullopt);
  // Add a dropout layer
  tensor_guid_t dropout(tensor_guid_t const &input,
                        float rate,
                        unsigned long long seed = 0,
                        std::optional<std::string> const &name = std::nullopt);
  // Add an embedding layer
  tensor_guid_t embedding(
      tensor_guid_t const &input,
      int num_entries,
      int outDim,
      AggregateOp aggr,
      DataType dtype = DataType::FLOAT,
      std::optional<InitializerAttrs> const &kernel_initializer = std::nullopt,
      std::optional<std::string> const &name = std::nullopt);
  // Add a gather layer
  tensor_guid_t gather(tensor_guid_t const &input,
                       tensor_guid_t const &index,
                       ff_dim_t dim,
                       std::optional<std::string> const &name = std::nullopt);
  // Add a cache layer
  tensor_guid_t
      cache(tensor_guid_t const &input,
            int num_batches,
            std::function<float(float *, void const *, void const *, int)>
                score_f = {},
            std::optional<std::string> const &name = std::nullopt);
  // Add a 2D pooling layer
  tensor_guid_t
      pool2d(tensor_guid_t const &input,
             int kernelH,
             int kernelW,
             int strideH,
             int strideW,
             int paddingH,
             int paddingW,
             PoolOp type = PoolOp::MAX,
             std::optional<Activation> const &activation = std::nullopt,
             std::optional<std::string> const &name = std::nullopt);
  tensor_guid_t
      layer_norm(tensor_guid_t const &input,
                 std::vector<int> const &axes,
                 bool elementwise_affine,
                 float eps,
                 std::optional<std::string> const &name = std::nullopt);
  tensor_guid_t
      batch_norm(tensor_guid_t const &input,
                 bool relu = true,
                 std::optional<std::string> const &name = std::nullopt);
  tensor_guid_t
      batch_matmul(tensor_guid_t const &A,
                   tensor_guid_t const &B,
                   int a_seq_length_dim = -1,
                   int b_seq_length_dim = -1,
                   std::optional<std::string> const &name = std::nullopt);
  tensor_guid_t dense(
      tensor_guid_t const &input,
      int outDim,
      std::optional<Activation> activation = std::nullopt,
      bool use_bias = true,
      DataType data_type = DataType::FLOAT,
      std::optional<InitializerAttrs> const &kernel_initializer = std::nullopt,
      std::optional<InitializerAttrs> const &bias_initializer = std::nullopt,
      std::optional<std::string> const &name = std::nullopt);
  // Add a cast layer
  tensor_guid_t cast(tensor_guid_t const &input,
                     DataType dtype,
                     std::optional<std::string> const &name = std::nullopt);
  // Add a concat layer
  tensor_guid_t
      concat(int n,
             std::vector<tensor_guid_t> const &tensors,
             int axis,
             std::optional<std::string> const &maybe_name = std::nullopt);
  // Add a mean layer
  tensor_guid_t mean(tensor_guid_t const &input,
                     std::vector<int> const &dims,
                     bool keepdims,
                     char const *name);
  // Add a split layer
  std::vector<tensor_guid_t>
      split(tensor_guid_t const &input,
            std::vector<int> const &split,
            int axis,
            std::optional<std::string> const &name = std::nullopt);
  // Add a flat layer
  tensor_guid_t flat(tensor_guid_t const &input,
                     std::optional<std::string> const &name = std::nullopt);
  // Add a softmax layer
  tensor_guid_t softmax(tensor_guid_t const &input,
                        std::optional<int> dim = std::nullopt,
                        std::optional<std::string> const &name = std::nullopt);
  // Create input tensors and constants
  tensor_guid_t
      transpose(tensor_guid_t const &input,
                std::vector<int> const &perm,
                std::optional<std::string> const &name = std::nullopt);
  tensor_guid_t
      reduce_sum(tensor_guid_t const &input,
                 std::vector<int> const &axes,
                 bool keepdims = false,
                 std::optional<std::string> const &name = std::nullopt);
  tensor_guid_t reshape(tensor_guid_t const &input,
                        std::vector<int> const &shape,
                        std::optional<std::string> const &name = std::nullopt);
  tensor_guid_t reverse(tensor_guid_t const &input,
                        int axis,
                        std::optional<std::string> const &name = std::nullopt);
  std::vector<tensor_guid_t>
      top_k(tensor_guid_t const &input,
            int k,
            bool sorted,
            std::optional<std::string> const &name = std::nullopt);
  tensor_guid_t multihead_attention(
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
      std::optional<InitializerAttrs> initializer = std::nullopt,
      std::optional<std::string> const &maybe_name = std::nullopt);
  tensor_guid_t create_tensor(TensorShape const &, CreateGrad);
  tensor_guid_t create_weight(
      TensorShape const &,
      bool create_grad = true,
      std::optional<InitializerAttrs> const &initializer = std::nullopt,
      std::optional<ParamSync> sync_type = std::nullopt);

  std::vector<tensor_guid_t> get_outputs(LayerAttrs const &) const;
  tensor_guid_t get_output(LayerAttrs const &, int idx) const;

  std::vector<tensor_guid_t> add_layer(LayerAttrs const &layer,
                                       std::vector<tensor_guid_t> const &inputs,
                                       std::vector<TensorAttrs> const &weights,
                                       std::vector<TensorAttrs> const &outputs);

private:
  TensorShape get_shape(tensor_guid_t const &) const;

  tensor_guid_t
      broadcast(tensor_guid_t const &, TensorDims const &, std::string const &);

  tensor_guid_t as_type(tensor_guid_t const &, DataType, std::string const &);

  tensor_guid_t add_layer(LayerAttrs const &layer,
                          std::vector<tensor_guid_t> const &inputs,
                          std::vector<TensorAttrs> const &weights,
                          TensorAttrs const &output);

  std::vector<tensor_guid_t> add_layer(LayerAttrs const &layer,
                                       std::vector<tensor_guid_t> const &inputs,
                                       std::vector<TensorAttrs> const &weights,
                                       std::vector<TensorShape> const &outputs);

  tensor_guid_t add_layer(LayerAttrs const &layer,
                          std::vector<tensor_guid_t> const &inputs,
                          std::vector<TensorAttrs> const &weights,
                          TensorShape const &output);

  TensorDims get_broadcast_target_dims(std::vector<tensor_guid_t> const &);
  TensorDims get_broadcast_target_dims(std::vector<TensorDims> const &);

  tensor_guid_t
      element_binary(OperatorType,
                     tensor_guid_t const &lhs,
                     tensor_guid_t const &rhs,
                     std::optional<std::string> const &name = std::nullopt);

  tensor_guid_t
      element_unary(OperatorType,
                    tensor_guid_t const &input,
                    std::optional<float> scalar,
                    std::optional<std::string> const &name = std::nullopt);

public:
  ComputationGraph computation_graph;
};

} // namespace FlexFlow

#endif
