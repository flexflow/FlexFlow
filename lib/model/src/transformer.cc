#include "model/transformer.h"

namespace FlexFlow {

Config::Config(void) {
  hidden_size = 1024;
  embedding_size = 1024;
  num_heads = 16;
  num_layers = 12;
  sequence_length = 512;
  batchSize = 8;
}

ComputationGraph create_computation_graph(Config &config) {
  ComputationGraphBuilder builder;
  // Create the t tensor
  std::vector<size_t> dims = {
      config.batchSize, config.sequence_length, config.hidden_size};
  DimOrdered<ff_dim_t, size_t> ff_ordered(dims);
  TensorDims t_dims(ff_ordered);
  tensor_guid_t t = builder.create_tensor(TensorShape(t_dims, DataType::FLOAT),
                                          CreateGrad::YES);

  for (int i = 0; i < config.num_layers; i++) {
    tensor_guid_t attention =
        builder.multihead_attention(t,
                                    t,
                                    t,
                                    config.hidden_size,
                                    config.num_heads,
                                    config.hidden_size / config.num_heads,
                                    config.hidden_size / config.num_heads,
                                    0.0f,
                                    true,
                                    false,
                                    false,
                                    {},
                                    "multihead_attention");
    tensor_guid_t dense1 = builder.dense(attention,
                                         config.hidden_size,
                                         Activation::RELU,
                                         false,
                                         DataType::FLOAT,
                                         std::nullopt,
                                         std::nullopt,
                                         "dense1");
    tensor_guid_t dense2 = builder.dense(dense1,
                                         config.hidden_size,
                                         std::nullopt,
                                         false,
                                         DataType::FLOAT,
                                         std::nullopt,
                                         std::nullopt,
                                         "dense2");
    t = dense2;
  }

  tensor_guid_t output = builder.dense(t,
                                       1,
                                       std::nullopt,
                                       false,
                                       DataType::FLOAT,
                                       std::nullopt,
                                       std::nullopt,
                                       "output_dense");
  return builder.computation_graph;
}
ParallelComputationGraph create_parallel_computation_graph(Config &config) {
  ParallelComputationGraphBuilder builder;
  // Create the input tensor

  ShardParallelDim dim(2, 4);
  std::vector<ShardParallelDim> dims = {dim, dim, dim};
  DimOrdered<ff_dim_t, ShardParallelDim> ff_ordered(dims);
  SumDegree sd(2);
  DiscardCopyDegree dcd(2);
  ReplicaParallelDimSet dims2(sd, dcd);
  ParallelTensorDims t_dims(ff_ordered, dims2);
  parallel_tensor_guid_t t = builder.create_input_tensor(
      ParallelTensorShape(t_dims, DataType::FLOAT), true, "input_tensor");

  for (int i = 0; i < config.num_layers; i++) {
    parallel_tensor_guid_t attention =
        builder.multihead_attention(t,
                                    t,
                                    t,
                                    config.hidden_size,
                                    config.num_heads,
                                    config.hidden_size / config.num_heads,
                                    config.hidden_size / config.num_heads,
                                    0.0f,
                                    true,
                                    false,
                                    false,
                                    {},
                                    {},
                                    {},
                                    "multihead_attention");
    parallel_tensor_guid_t fused_dense = builder.dense(attention,
                                                       config.hidden_size,
                                                       Activation::RELU,
                                                       false,
                                                       DataType::FLOAT,
                                                       std::nullopt,
                                                       std::nullopt,
                                                       "fused_dense");
    t = fused_dense;
  }

  parallel_tensor_guid_t output = builder.dense(t,
                                                1,
                                                std::nullopt,
                                                false,
                                                DataType::FLOAT,
                                                std::nullopt,
                                                std::nullopt,
                                                "output_dense");
  return builder.pcg;
}
} // namespace FlexFlow
