#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_COMPUTATION_GRAPH_BUILDER_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_COMPUTATION_GRAPH_BUILDER_H

#include "op-attrs/upsample_mode.dtg.h"
#include "pcg/computation_graph.dtg.h"
#include "pcg/tensor_guid_t.dtg.h"

namespace FlexFlow {

/**
 * \brief A helper interface for building ComputationGraph in a pytorch (i.e.,
 * weight-implicit) style.
 *
 * For an example of how to use it, see the following code from \ref models "":
 * \snippet lib/models/src/models/split_test/split_test.cc ComputationGraphBuilder example
 */
struct ComputationGraphBuilder {
public:
  ComputationGraphBuilder();

  tensor_guid_t exp(tensor_guid_t const &,
                    std::optional<std::string> const &name = std::nullopt);

  tensor_guid_t add(tensor_guid_t const &lhs,
                    tensor_guid_t const &rhs,
                    std::optional<std::string> const &name = std::nullopt);

  tensor_guid_t subtract(tensor_guid_t const &x,
                         tensor_guid_t const &y,
                         std::optional<std::string> const &name = std::nullopt);

  tensor_guid_t multiply(tensor_guid_t const &x,
                         tensor_guid_t const &y,
                         std::optional<std::string> const &name = std::nullopt);

  tensor_guid_t divide(tensor_guid_t const &x,
                       tensor_guid_t const &y,
                       std::optional<std::string> const &name = std::nullopt);

  tensor_guid_t max(tensor_guid_t const &x,
                    tensor_guid_t const &y,
                    std::optional<std::string> const &name = std::nullopt);

  tensor_guid_t min(tensor_guid_t const &x,
                    tensor_guid_t const &y,
                    std::optional<std::string> const &name = std::nullopt);

  tensor_guid_t rsqrt(tensor_guid_t const &x,
                      std::optional<std::string> const &name = std::nullopt);

  tensor_guid_t pow(tensor_guid_t const &x,
                    float exponent,
                    std::optional<std::string> const &name = std::nullopt);

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

  tensor_guid_t sin(tensor_guid_t const &x,
                    std::optional<std::string> const &name = std::nullopt);

  tensor_guid_t cos(tensor_guid_t const &x,
                    std::optional<std::string> const &name = std::nullopt);

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

  tensor_guid_t silu(tensor_guid_t const &x,
                     std::optional<std::string> const &name = std::nullopt);

  tensor_guid_t conv2d(
      tensor_guid_t const &input,
      positive_int outChannels,
      positive_int kernelH,
      positive_int kernelW,
      positive_int strideH,
      positive_int strideW,
      nonnegative_int paddingH,
      nonnegative_int paddingW,
      std::optional<Activation> const &activation = std::nullopt,
      positive_int groups = 1_p,
      bool use_bias = true,
      std::optional<InitializerAttrs> const &kernel_initializer = std::nullopt,
      std::optional<InitializerAttrs> const &bias_initializer = std::nullopt,
      std::optional<RegularizerAttrs> const &kernel_regularizer = std::nullopt,
      std::optional<std::string> const &name = std::nullopt);

  tensor_guid_t upsample(tensor_guid_t const &input,
                         int_ge_two scale_factor,
                         UpsampleMode mode = UpsampleMode::NEAREST,
                         std::optional<std::string> const &name = std::nullopt);

  tensor_guid_t dropout(tensor_guid_t const &input,
                        float rate,
                        unsigned long long seed = 0,
                        std::optional<std::string> const &name = std::nullopt);

  tensor_guid_t embedding(
      tensor_guid_t const &input,
      positive_int num_entries,
      positive_int outDim,
      AggregateOp aggr,
      DataType dtype = DataType::FLOAT,
      std::optional<InitializerAttrs> const &initializer = std::nullopt,
      std::optional<std::string> const &name = std::nullopt);

  tensor_guid_t gather(tensor_guid_t const &input,
                       tensor_guid_t const &index,
                       relative_ff_dim_t dim,
                       std::optional<std::string> const &name = std::nullopt);

  tensor_guid_t
      cache(tensor_guid_t const &input,
            nonnegative_int num_batches,
            std::function<float(float *, void const *, void const *, int)>
                score_f = {},
            std::optional<std::string> const &name = std::nullopt);

  tensor_guid_t
      pool2d(tensor_guid_t const &input,
             positive_int kernelH,
             positive_int kernelW,
             positive_int strideH,
             positive_int strideW,
             nonnegative_int paddingH,
             nonnegative_int paddingW,
             PoolOp type = PoolOp::MAX,
             std::optional<Activation> const &activation = std::nullopt,
             std::optional<std::string> const &name = std::nullopt);

  tensor_guid_t adaptive_pool2d(
      tensor_guid_t const &input,
      positive_int output_h,
      positive_int output_w,
      PoolOp type = PoolOp::MAX,
      std::optional<Activation> const &activation = std::nullopt,
      std::optional<std::string> const &name = std::nullopt);

  tensor_guid_t
      layer_norm(tensor_guid_t const &input,
                 std::set<relative_ff_dim_t> const &axes,
                 bool elementwise_affine,
                 float eps,
                 std::optional<std::string> const &name = std::nullopt);

  tensor_guid_t
      batch_norm(tensor_guid_t const &input,
                 bool affine = true,
                 std::optional<Activation> const &activation = std::nullopt,
                 float eps = 1e-5f,
                 std::optional<float> const &momentum = 0.1f,
                 std::optional<std::string> const &name = std::nullopt);

  tensor_guid_t
      batch_matmul(tensor_guid_t const &lhs,
                   tensor_guid_t const &rhs,
                   std::optional<std::string> const &name = std::nullopt);

  tensor_guid_t dense(
      tensor_guid_t const &input,
      positive_int outDim,
      std::optional<Activation> activation = std::nullopt,
      bool use_bias = true,
      DataType data_type = DataType::FLOAT,
      std::optional<InitializerAttrs> const &projection_initializer =
          std::nullopt,
      std::optional<InitializerAttrs> const &bias_initializer = std::nullopt,
      std::optional<std::string> const &name = std::nullopt);

  tensor_guid_t cast(tensor_guid_t const &input,
                     DataType dtype,
                     std::optional<std::string> const &name = std::nullopt);

  tensor_guid_t concat(std::vector<tensor_guid_t> const &tensors,
                       relative_ff_dim_t axis,
                       std::optional<std::string> const &name = std::nullopt);

  tensor_guid_t mean(tensor_guid_t const &input,
                     std::vector<int> const &dims,
                     bool keepdims,
                     char const *name);

  std::vector<tensor_guid_t>
      split(tensor_guid_t const &input,
            std::vector<positive_int> const &split,
            relative_ff_dim_t axis,
            std::optional<std::string> const &name = std::nullopt);

  tensor_guid_t
      flat(tensor_guid_t const &input,
           relative_ff_dim_t start_dim = relative_ff_dim_t{0},
           std::optional<relative_ff_dim_t> const &end_dim = std::nullopt,
           std::optional<std::string> const &name = std::nullopt);

  tensor_guid_t softmax(tensor_guid_t const &input,
                        std::optional<relative_ff_dim_t> dim = std::nullopt,
                        std::optional<std::string> const &name = std::nullopt);

  tensor_guid_t
      transpose(tensor_guid_t const &input,
                std::vector<nonnegative_int> const &perm,
                std::optional<std::string> const &name = std::nullopt);

  tensor_guid_t
      reduce_sum(tensor_guid_t const &input,
                 std::vector<relative_ff_dim_t> const &axes,
                 bool keepdims = false,
                 std::optional<std::string> const &name = std::nullopt);

  tensor_guid_t reshape(tensor_guid_t const &input,
                        std::vector<positive_int> const &shape,
                        std::optional<std::string> const &name = std::nullopt);

  tensor_guid_t reverse(tensor_guid_t const &input,
                        relative_ff_dim_t axis,
                        std::optional<std::string> const &name = std::nullopt);

  std::vector<tensor_guid_t>
      top_k(tensor_guid_t const &input,
            nonnegative_int k,
            bool sorted,
            std::optional<std::string> const &name = std::nullopt);

  tensor_guid_t multihead_attention(
      tensor_guid_t const &query,
      tensor_guid_t const &key,
      tensor_guid_t const &value,
      positive_int embed_dim,
      positive_int num_heads,
      std::optional<positive_int> const &kdim = std::nullopt,
      std::optional<positive_int> const &vdim = std::nullopt,
      float dropout = 0.0f,
      bool bias = true,
      bool add_bias_kv = false,
      bool add_zero_attn = false,
      std::optional<InitializerAttrs> initializer = std::nullopt,
      std::optional<std::string> const &maybe_name = std::nullopt);

  tensor_guid_t
      create_input(TensorShape const &,
                   CreateGrad,
                   std::optional<std::string> const &name = std::nullopt);

  tensor_guid_t
      create_weight(TensorShape const &shape,
                    InitializerAttrs const &initializer,
                    std::optional<std::string> const &name = std::nullopt);
  tensor_guid_t
      create_weight(TensorAttrs const &,
                    std::optional<std::string> const &name = std::nullopt);

  std::vector<tensor_guid_t> get_outputs(LayerAttrs const &) const;
  tensor_guid_t get_output(LayerAttrs const &, nonnegative_int idx) const;

  TensorShape get_shape(tensor_guid_t const &) const;

private:
  std::map<TensorSlotName, tensor_guid_t> add_layer(
      LayerAttrs const &layer,
      std::map<TensorSlotName, tensor_guid_t> const &inputs,
      std::map<TensorSlotName, InitializerAttrs> const &weights,
      std::optional<std::map<TensorSlotName, CreateGrad>> const &outputs =
          std::nullopt);

  tensor_guid_t
      broadcast(tensor_guid_t const &, TensorDims const &, std::string const &);

  tensor_guid_t as_type(tensor_guid_t const &, DataType, std::string const &);

  TensorDims get_broadcast_target_dims(std::vector<tensor_guid_t> const &);
  TensorDims get_broadcast_target_dims(std::vector<TensorDims> const &);

  tensor_guid_t
      element_binary(ElementBinaryOp,
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
