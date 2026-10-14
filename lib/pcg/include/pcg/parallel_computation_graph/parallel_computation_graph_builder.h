#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_PARALLEL_COMPUTATION_GRAPH_BUILDER_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_PARALLEL_COMPUTATION_GRAPH_BUILDER_H

#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.dtg.h"
#include <optional>

namespace FlexFlow {

struct ParallelComputationGraphBuilder {
public:
  ParallelComputationGraphBuilder();

  parallel_tensor_guid_t create_input_tensor(
      TensorShape const &shape,
      std::optional<std::string> const &name = std::nullopt);

  parallel_tensor_guid_t
      add(parallel_tensor_guid_t const &lhs,
          parallel_tensor_guid_t const &rhs,
          std::optional<std::string> const &name = std::nullopt);

  parallel_tensor_guid_t
      cast(parallel_tensor_guid_t const &input,
           DataType result_type,
           std::optional<std::string> const &name = std::nullopt);

  parallel_tensor_guid_t conv2d(
      parallel_tensor_guid_t const &input,
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

  parallel_tensor_guid_t dense(
      parallel_tensor_guid_t const &input,
      positive_int outDim,
      std::optional<Activation> activation = std::nullopt,
      bool use_bias = true,
      DataType data_type = DataType::FLOAT,
      std::optional<InitializerAttrs> const &projection_initializer =
          std::nullopt,
      std::optional<InitializerAttrs> const &bias_initializer = std::nullopt,
      std::optional<std::string> const &name = std::nullopt);

  parallel_tensor_guid_t embedding(
      parallel_tensor_guid_t const &input,
      positive_int num_entries,
      positive_int outDim,
      AggregateOp aggr,
      DataType dtype = DataType::FLOAT,
      std::optional<InitializerAttrs> const &kernel_initializer = std::nullopt,
      std::optional<std::string> const &name = std::nullopt);

  parallel_tensor_guid_t multihead_attention(
      parallel_tensor_guid_t const &query,
      parallel_tensor_guid_t const &key,
      parallel_tensor_guid_t const &value,
      positive_int embed_dim,
      positive_int num_heads,
      std::optional<positive_int> kdim = std::nullopt,
      std::optional<positive_int> vdim = std::nullopt,
      float dropout = 0.0f,
      bool bias = true,
      bool add_bias_kv = false,
      bool add_zero_attn = false,
      std::optional<InitializerAttrs> initializer = std::nullopt,
      std::optional<InitializerAttrs> input_bias_initializer = std::nullopt,
      std::optional<InitializerAttrs> output_bias_initializer = std::nullopt,
      std::optional<std::string> const &name = std::nullopt);

  parallel_tensor_guid_t
      batch_norm(parallel_tensor_guid_t const &input,
                 bool affine,
                 std::optional<Activation> const &activation,
                 float eps,
                 std::optional<float> const &momentum,
                 std::optional<std::string> const &name = std::nullopt);

  parallel_tensor_guid_t
      relu(parallel_tensor_guid_t const &x,
           std::optional<std::string> const &name = std::nullopt);

  parallel_tensor_guid_t
      identity(parallel_tensor_guid_t const &x,
               std::optional<std::string> const &name = std::nullopt);

  parallel_tensor_guid_t
      gelu(parallel_tensor_guid_t const &x,
           std::optional<std::string> const &name = std::nullopt);

  parallel_tensor_guid_t
      sigmoid(parallel_tensor_guid_t const &x,
              std::optional<std::string> const &name = std::nullopt);

  parallel_tensor_guid_t
      tanh(parallel_tensor_guid_t const &x,
           std::optional<std::string> const &name = std::nullopt);

  parallel_tensor_guid_t
      elu(parallel_tensor_guid_t const &x,
          std::optional<std::string> const &name = std::nullopt);

  parallel_tensor_guid_t
      silu(parallel_tensor_guid_t const &x,
           std::optional<std::string> const &name = std::nullopt);

  parallel_tensor_guid_t
      parallel_partition(parallel_tensor_guid_t const &input,
                         ff_dim_t dim,
                         int_ge_two degree,
                         std::optional<std::string> const &name = std::nullopt);
  parallel_tensor_guid_t
      parallel_combine(parallel_tensor_guid_t const &x,
                       ff_dim_t dim,
                       int_ge_two degree,
                       std::optional<std::string> const &name = std::nullopt);
  parallel_tensor_guid_t
      parallel_replicate(parallel_tensor_guid_t const &x,
                         int_ge_two degree,
                         std::optional<std::string> const &name = std::nullopt);
  parallel_tensor_guid_t
      parallel_reduce(parallel_tensor_guid_t const &x,
                      int_ge_two degree,
                      std::optional<std::string> const &name = std::nullopt);

  ParallelTensorShape get_shape(parallel_tensor_guid_t const &) const;

private:
  parallel_tensor_guid_t as_type(parallel_tensor_guid_t const &,
                                 DataType,
                                 std::string const &name);

private:
  std::map<TensorSlotName, parallel_tensor_guid_t> add_layer(
      ParallelLayerAttrs const &layer,
      std::map<TensorSlotName, parallel_tensor_guid_t> const &inputs,
      std::map<TensorSlotName, InitializerAttrs> const &weight_initializers);

  parallel_tensor_guid_t
      add_weight(ParallelTensorShape const &weight_tensor_shape,
                 InitializerAttrs const &initializer,
                 std::optional<std::string> const &name = std::nullopt);

  parallel_tensor_guid_t
      element_unary(ElementUnaryAttrs const &element_unary_attrs,
                    parallel_tensor_guid_t const &input,
                    std::optional<std::string> const &name);

public:
  ParallelComputationGraph pcg;
};

} // namespace FlexFlow

#endif
