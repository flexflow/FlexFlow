#include "model/ResNet.h"

namespace FlexFlow {

tensor_guid_t BottleneckBlock(ComputationGraphBuilder &builder,
                              tensor_guid_t input,
                              int out_channels,
                              int stride) {
  tensor_guid_t t = builder.conv2d(input,
                                   out_channels,
                                   1,
                                   1,
                                   1,
                                   1,
                                   0,
                                   0,
                                   std::nullopt,
                                   1,
                                   true,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   "conv1");
  t = builder.conv2d(t,
                     out_channels,
                     3,
                     3,
                     stride,
                     stride,
                     1,
                     1,
                     std::nullopt,
                     1,
                     true,
                     std::nullopt,
                     std::nullopt,
                     std::nullopt,
                     "conv2");
  t = builder.conv2d(t,
                     4 * out_channels,
                     1,
                     1,
                     1,
                     1,
                     0,
                     0,
                     std::nullopt,
                     1,
                     true,
                     std::nullopt,
                     std::nullopt,
                     std::nullopt,
                     "conv3");

  // if case ignored for now

  t = builder.add(input, t, "add");
  t = builder.relu(t, "relu");
  return t;
}

parallel_tensor_guid_t BottleneckBlock(ParallelComputationGraphBuilder &builder,
                                       parallel_tensor_guid_t input,
                                       int out_channels,
                                       int stride) {
  parallel_tensor_guid_t t = builder.conv2d(input,
                                            out_channels,
                                            1,
                                            1,
                                            1,
                                            1,
                                            0,
                                            0,
                                            std::nullopt,
                                            1,
                                            true,
                                            std::nullopt,
                                            std::nullopt,
                                            std::nullopt,
                                            "conv1");
  t = builder.conv2d(t,
                     out_channels,
                     3,
                     3,
                     stride,
                     stride,
                     1,
                     1,
                     std::nullopt,
                     1,
                     true,
                     std::nullopt,
                     std::nullopt,
                     std::nullopt,
                     "conv2");
  t = builder.conv2d(t,
                     4 * out_channels,
                     1,
                     1,
                     1,
                     1,
                     0,
                     0,
                     std::nullopt,
                     1,
                     true,
                     std::nullopt,
                     std::nullopt,
                     std::nullopt,
                     "conv3");

  // if case ignored for now

  t = builder.add(input, t, "add");
  t = builder.relu(t, "relu");
  return t;
}

ComputationGraph ResNet::create_computation_graph(Config &config) {
  ComputationGraphBuilder builder;
  // Create the t tensor
  std::vector<size_t> dims = {config.batchSize, 3, 229, 229};
  DimOrdered<ff_dim_t, size_t> ff_ordered(dims);
  TensorDims t_dims(ff_ordered);
  tensor_guid_t t = builder.create_tensor(TensorShape(t_dims, DataType::FLOAT),
                                          CreateGrad::YES);

  t = builder.conv2d(t,
                     64,
                     7,
                     7,
                     2,
                     2,
                     3,
                     3,
                     std::nullopt,
                     1,
                     true,
                     std::nullopt,
                     std::nullopt,
                     std::nullopt,
                     "conv1");
  t = builder.pool2d(t, 3, 3, 2, 2, 1, 1, PoolOp::MAX, std::nullopt, "pool1");
  for (int i = 0; i < 3; i++) {
    t = BottleneckBlock(builder, t, 64, 1);
  }
  for (int i = 0; i < 4; i++) {
    int stride = (i == 0) ? 2 : 1;
    t = BottleneckBlock(builder, t, 128, stride);
  }
  for (int i = 0; i < 6; i++) {
    int stride = (i == 0) ? 2 : 1;
    t = BottleneckBlock(builder, t, 256, stride);
  }
  for (int i = 0; i < 3; i++) {
    int stride = (i == 0) ? 2 : 1;
    t = BottleneckBlock(builder, t, 512, stride);
  }
  t = builder.pool2d(t, 7, 7, 1, 1, 0, 0, PoolOp::AVG, std::nullopt, "pool2");
  t = builder.flat(t, "flat");
  t = builder.dense(t,
                    10,
                    std::nullopt,
                    true,
                    DataType::FLOAT,
                    std::nullopt,
                    std::nullopt,
                    "dense");
  t = builder.softmax(t, -1, "softmax");

  return builder.computation_graph;
}

ParallelComputationGraph ResNet::create_parallel_computation_graph() {
  ParallelComputationGraphBuilder builder;
  ShardParallelDim dim(2, 4);
  std::vector<ShardParallelDim> dims = {dim, dim, dim};
  DimOrdered<ff_dim_t, ShardParallelDim> ff_ordered(dims);
  SumDegree sd(2);
  DiscardCopyDegree dcd(2);
  ReplicaParallelDimSet dims2(sd, dcd);
  ParallelTensorDims t_dims(ff_ordered, dims2);
  parallel_tensor_guid_t t = builder.create_input_tensor(
      ParallelTensorShape(t_dims, DataType::FLOAT), true, "input_tensor");

  t = builder.conv2d(t,
                     64,
                     7,
                     7,
                     2,
                     2,
                     3,
                     3,
                     std::nullopt,
                     1,
                     true,
                     std::nullopt,
                     std::nullopt,
                     std::nullopt,
                     "conv1");
  t = builder.pool2d(
      t, 3, 3, 2, 2, 1, 1, PoolOp::MAX, Activation::RELU, "pool1");
  for (int i = 0; i < 3; i++) {
    t = BottleneckBlock(builder, t, 64, 1);
  }
  for (int i = 0; i < 4; i++) {
    int stride = (i == 0) ? 2 : 1;
    t = BottleneckBlock(builder, t, 128, stride);
  }
  for (int i = 0; i < 6; i++) {
    int stride = (i == 0) ? 2 : 1;
    t = BottleneckBlock(builder, t, 256, stride);
  }
  for (int i = 0; i < 3; i++) {
    int stride = (i == 0) ? 2 : 1;
    t = BottleneckBlock(builder, t, 512, stride);
  }
  t = builder.pool2d(
      t, 7, 7, 1, 1, 0, 0, PoolOp::AVG, Activation::RELU, "pool2");
  t = builder.flat(t, "flat");
  t = builder.dense(t,
                    10,
                    std::nullopt,
                    true,
                    DataType::FLOAT,
                    std::nullopt,
                    std::nullopt,
                    "dense");
  t = builder.softmax(t, -1, "softmax");

  return builder.pcg;
}

} // namespace FlexFlow
