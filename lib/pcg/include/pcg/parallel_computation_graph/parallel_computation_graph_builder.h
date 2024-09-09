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
      ParallelTensorShape const &shape,
      CreateGrad create_grad = CreateGrad::YES,
      std::optional<std::string> const &name = std::nullopt);

  parallel_tensor_guid_t
      add(parallel_tensor_guid_t const &lhs,
          parallel_tensor_guid_t const &rhs,
          std::optional<std::string> const &name = std::nullopt);

  parallel_tensor_guid_t
      batch_matmul(parallel_tensor_guid_t const &a,
                   parallel_tensor_guid_t const &b,
                   std::optional<std::string> const &name = std::nullopt);

  parallel_tensor_guid_t
      cast(parallel_tensor_guid_t const &input,
           DataType result_type,
           std::optional<std::string> const &name = std::nullopt);

  parallel_tensor_guid_t conv2d(
      parallel_tensor_guid_t const &input,
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

  parallel_tensor_guid_t dense(
      parallel_tensor_guid_t const &input,
      int outDim,
      std::optional<Activation> activation = std::nullopt,
      bool use_bias = true,
      DataType data_type = DataType::FLOAT,
      std::optional<InitializerAttrs> const &projection_initializer =
          std::nullopt,
      std::optional<InitializerAttrs> const &bias_initializer = std::nullopt,
      std::optional<std::string> const &name = std::nullopt);

  parallel_tensor_guid_t embedding(
      parallel_tensor_guid_t const &input,
      int num_entries,
      int outDim,
      AggregateOp aggr,
      DataType dtype = DataType::FLOAT,
      std::optional<InitializerAttrs> const &kernel_initializer = std::nullopt,
      std::optional<std::string> const &name = std::nullopt);

  parallel_tensor_guid_t multihead_attention(
      parallel_tensor_guid_t const &query,
      parallel_tensor_guid_t const &key,
      parallel_tensor_guid_t const &value,
      int embed_dim,
      int num_heads,
      std::optional<int> kdim = std::nullopt,
      std::optional<int> vdim = std::nullopt,
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
                 bool relu = true,
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
      parallel_partition(parallel_tensor_guid_t const &x,
                         ff_dim_t dim,
                         int degree,
                         std::optional<std::string> const &name = std::nullopt);
  parallel_tensor_guid_t
      parallel_combine(parallel_tensor_guid_t const &x,
                       ff_dim_t dim,
                       int degree,
                       std::optional<std::string> const &name = std::nullopt);
  parallel_tensor_guid_t
      parallel_replicate(parallel_tensor_guid_t const &x,
                         int degree,
                         std::optional<std::string> const &name = std::nullopt);
  parallel_tensor_guid_t
      parallel_reduce(parallel_tensor_guid_t const &x,
                      int degree,
                      std::optional<std::string> const &name = std::nullopt);

private:
  parallel_tensor_guid_t as_type(parallel_tensor_guid_t const &,
                                 DataType,
                                 std::string const &name);

private:
  ParallelTensorShape get_shape(parallel_tensor_guid_t const &) const;

private:
  std::vector<parallel_tensor_guid_t>
      add_layer(ParallelLayerAttrs const &layer,
                std::vector<parallel_tensor_guid_t> const &inputs,
                std::vector<ParallelTensorAttrs> const &weights,
                std::vector<ParallelTensorAttrs> const &outputs);
  std::vector<parallel_tensor_guid_t>
      add_layer(ParallelLayerAttrs const &layer,
                std::vector<parallel_tensor_guid_t> const &inputs,
                std::vector<ParallelTensorAttrs> const &weights,
                std::vector<ParallelTensorShape> const &output);
  parallel_tensor_guid_t
      add_layer(ParallelLayerAttrs const &layer,
                std::vector<parallel_tensor_guid_t> const &inputs,
                std::vector<ParallelTensorAttrs> const &weights,
                ParallelTensorAttrs const &output);
  parallel_tensor_guid_t
      add_layer(ParallelLayerAttrs const &layer,
                std::vector<parallel_tensor_guid_t> const &inputs,
                std::vector<ParallelTensorAttrs> const &weights,
                ParallelTensorShape const &output);

  parallel_tensor_guid_t
      add_weight(ParallelTensorAttrs const &weight_tensor_attrs,
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
