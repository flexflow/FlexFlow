#include "models/inceptionv3.h"
#include "pcg/computation_graph.h"
#include "pcg/computation_graph_builder.h"

namespace FlexFlow {

InceptionV3Config get_default_inception_v3_config() {
  return InceptionV3Config{/*input_height=*/299,
                           /*input_width=*/299,
                           /*input_num_channels=*/3,
                           /*num_classes=*/1000,
                           /*batch_size=*/32};
}

tensor_guid_t create_conv_block(ComputationGraphBuilder &cgb,
                                tensor_guid_t const &input,
                                int filters,
                                int kernel_size_h,
                                int kernel_size_w,
                                int stride_h = 1,
                                int stride_w = 1,
                                int padding_h = 0,
                                int padding_w = 0,
                                bool use_bias = false) {
  tensor_guid_t conv = cgb.conv2d(input,
                                  filters,
                                  kernel_size_h,
                                  kernel_size_w,
                                  stride_h,
                                  stride_w,
                                  padding_h,
                                  padding_w,
                                  std::nullopt,
                                  1,
                                  use_bias);
  return cgb.batch_norm(conv);
}

tensor_guid_t create_inception_module_a(ComputationGraphBuilder &cgb,
                                        tensor_guid_t const &input,
                                        int pool_features) {
  tensor_guid_t branch1x1 = create_conv_block(cgb, input, 64, 1, 1);

  tensor_guid_t branch5x5 = create_conv_block(cgb, input, 48, 1, 1);
  branch5x5 = create_conv_block(cgb, branch5x5, 64, 5, 5, 1, 1, 2, 2);

  tensor_guid_t branch3x3dbl = create_conv_block(cgb, input, 64, 1, 1);
  branch3x3dbl = create_conv_block(cgb, branch3x3dbl, 96, 3, 3, 1, 1, 1, 1);
  branch3x3dbl = create_conv_block(cgb, branch3x3dbl, 96, 3, 3, 1, 1, 1, 1);

  tensor_guid_t branch_pool = cgb.pool2d(input, 3, 3, 1, 1, 1, 1, PoolOp::AVG);
  branch_pool = create_conv_block(cgb, branch_pool, pool_features, 1, 1);

  return cgb.concat(4, {branch1x1, branch5x5, branch3x3dbl, branch_pool}, 3);
}

tensor_guid_t create_inception_module_b(ComputationGraphBuilder &cgb,
                                        tensor_guid_t const &input) {
  tensor_guid_t branch3x3 = create_conv_block(cgb, input, 384, 3, 3, 2, 2);

  tensor_guid_t branch3x3dbl = create_conv_block(cgb, input, 64, 1, 1);
  branch3x3dbl = create_conv_block(cgb, branch3x3dbl, 96, 3, 3, 1, 1, 1, 1);
  branch3x3dbl = create_conv_block(cgb, branch3x3dbl, 96, 3, 3, 2, 2);

  tensor_guid_t branch_pool = cgb.pool2d(input, 3, 3, 2, 2, 0, 0, PoolOp::MAX);

  return cgb.concat(3, {branch3x3, branch3x3dbl, branch_pool}, 3);
}

tensor_guid_t create_inception_module_c(ComputationGraphBuilder &cgb,
                                        tensor_guid_t const &input,
                                        int channels_7x7) {
  tensor_guid_t branch1x1 = create_conv_block(cgb, input, 192, 1, 1);

  tensor_guid_t branch7x7 = create_conv_block(cgb, input, channels_7x7, 1, 1);
  branch7x7 = create_conv_block(cgb, branch7x7, channels_7x7, 1, 7, 1, 1, 0, 3);
  branch7x7 = create_conv_block(cgb, branch7x7, 192, 7, 1, 1, 1, 3, 0);

  tensor_guid_t branch7x7dbl =
      create_conv_block(cgb, input, channels_7x7, 1, 1);
  branch7x7dbl =
      create_conv_block(cgb, branch7x7dbl, channels_7x7, 7, 1, 1, 1, 3, 0);
  branch7x7dbl =
      create_conv_block(cgb, branch7x7dbl, channels_7x7, 1, 7, 1, 1, 0, 3);
  branch7x7dbl =
      create_conv_block(cgb, branch7x7dbl, channels_7x7, 7, 1, 1, 1, 3, 0);
  branch7x7dbl =
      create_conv_block(cgb, branch7x7dbl, channels_7x7, 1, 7, 1, 1, 0, 3);

  tensor_guid_t branch_pool = cgb.pool2d(input, 3, 3, 1, 1, 1, 1, PoolOp::AVG);
  branch_pool = create_conv_block(cgb, branch_pool, 192, 1, 1);

  return cgb.concat(4, {branch1x1, branch7x7, branch7x7dbl, branch_pool}, 3);
}

tensor_guid_t create_inception_module_d(ComputationGraphBuilder &cgb,
                                        tensor_guid_t const &input) {
  tensor_guid_t branch3x3 = create_conv_block(cgb, input, 192, 1, 1);
  branch3x3 = create_conv_block(cgb, branch3x3, 320, 3, 3, 2, 2);

  tensor_guid_t branch7x7x3 = create_conv_block(cgb, input, 192, 1, 1);
  branch7x7x3 = create_conv_block(cgb, branch7x7x3, 192, 1, 7, 1, 1, 0, 3);
  branch7x7x3 = create_conv_block(cgb, branch7x7x3, 192, 7, 1, 1, 1, 3, 0);
  branch7x7x3 = create_conv_block(cgb, branch7x7x3, 192, 3, 3, 2, 2);

  tensor_guid_t branch_pool = cgb.pool2d(input, 3, 3, 2, 2, 0, 0, PoolOp::MAX);

  return cgb.concat(3, {branch3x3, branch7x7x3, branch_pool}, 3);
}

tensor_guid_t create_inception_module_e(ComputationGraphBuilder &cgb,
                                        tensor_guid_t const &input) {
  tensor_guid_t branch1x1 = create_conv_block(cgb, input, 320, 1, 1);

  tensor_guid_t branch3x3 = create_conv_block(cgb, input, 384, 1, 1);
  tensor_guid_t branch3x3_1 =
      create_conv_block(cgb, branch3x3, 384, 1, 3, 1, 1, 0, 1);
  tensor_guid_t branch3x3_2 =
      create_conv_block(cgb, branch3x3, 384, 3, 1, 1, 1, 1, 0);
  branch3x3 = cgb.concat(2, {branch3x3_1, branch3x3_2}, 3);

  tensor_guid_t branch3x3dbl = create_conv_block(cgb, input, 448, 1, 1);
  branch3x3dbl = create_conv_block(cgb, branch3x3dbl, 384, 3, 3, 1, 1, 1, 1);
  tensor_guid_t branch3x3dbl_1 =
      create_conv_block(cgb, branch3x3dbl, 384, 1, 3, 1, 1, 0, 1);
  tensor_guid_t branch3x3dbl_2 =
      create_conv_block(cgb, branch3x3dbl, 384, 3, 1, 1, 1, 1, 0);
  branch3x3dbl = cgb.concat(2, {branch3x3dbl_1, branch3x3dbl_2}, 3);

  tensor_guid_t branch_pool = cgb.pool2d(input, 3, 3, 1, 1, 1, 1, PoolOp::AVG);
  branch_pool = create_conv_block(cgb, branch_pool, 192, 1, 1);

  return cgb.concat(4, {branch1x1, branch3x3, branch3x3dbl, branch_pool}, 3);
}

tensor_guid_t create_initial_layers(ComputationGraphBuilder &cgb,
                                    tensor_guid_t const &input) {
  tensor_guid_t x = create_conv_block(cgb, input, 32, 3, 3, 2, 2);
  x = create_conv_block(cgb, x, 32, 3, 3);
  x = create_conv_block(cgb, x, 64, 3, 3, 1, 1, 1, 1);
  x = cgb.pool2d(x, 3, 3, 2, 2, 0, 0, PoolOp::MAX);

  x = create_conv_block(cgb, x, 80, 1, 1);
  x = create_conv_block(cgb, x, 192, 3, 3);
  x = cgb.pool2d(x, 3, 3, 2, 2, 0, 0, PoolOp::MAX);

  return x;
}

tensor_guid_t create_final_layers(ComputationGraphBuilder &cgb,
                                  tensor_guid_t const &input,
                                  size_t num_classes) {
  tensor_guid_t x = cgb.pool2d(input, 8, 8, 1, 1, 0, 0, PoolOp::AVG);
  x = cgb.dropout(x, 0.5);
  x = cgb.dense(x, num_classes);
  return x;
}

tensor_guid_t create_inception_v3(ComputationGraphBuilder &cgb,
                                  InceptionV3Config const &config,
                                  tensor_guid_t const &input) {
  tensor_guid_t x = create_initial_layers(cgb, input);

  x = create_inception_module_a(cgb, x, 32);
  x = create_inception_module_a(cgb, x, 64);
  x = create_inception_module_a(cgb, x, 64);

  x = create_inception_module_b(cgb, x);

  x = create_inception_module_c(cgb, x, 128);
  x = create_inception_module_c(cgb, x, 160);
  x = create_inception_module_c(cgb, x, 160);
  x = create_inception_module_c(cgb, x, 192);

  x = create_inception_module_d(cgb, x);

  x = create_inception_module_e(cgb, x);
  x = create_inception_module_e(cgb, x);

  x = create_final_layers(cgb, x, config.num_classes);

  return x;
}

ComputationGraph
    get_inception_v3_computation_graph(InceptionV3Config const &config) {
  ComputationGraphBuilder cgb;

  TensorShape input_shape = TensorShape{
      TensorDims{FFOrdered<size_t>{config.batch_size,
                                   config.input_height,
                                   config.input_width,
                                   config.input_num_channels}},
      DataType::FLOAT,
  };

  tensor_guid_t input = cgb.create_tensor(input_shape, CreateGrad::YES);
  tensor_guid_t output = create_inception_v3(cgb, config, input);

  return cgb.computation_graph;
}

} // namespace FlexFlow
