/**
 * @file nasnet.cc
 * @brief Implement NASNet-A Large.
 */

#include "models/nasnet/nasnet.h"
#include "op-attrs/tensor_dims.h"
#include "op-attrs/tensor_shape.h"
#include "pcg/computation_graph_builder.h"

#include <algorithm>
#include <fmt/format.h>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace FlexFlow {

namespace {

struct SpatialPadding {
  int top;
  int bottom;
  int left;
  int right;
};

static std::pair<int, int>
    get_same_padding_for_dim(int input_size, int kernel_size, int stride) {
  int output_size = (input_size + stride - 1) / stride;
  int total_padding =
      std::max((output_size - 1) * stride + kernel_size - input_size, 0);
  int padding_before = total_padding / 2;
  return {padding_before, total_padding - padding_before};
}

static SpatialPadding get_same_padding(ComputationGraphBuilder const &cgb,
                                       tensor_guid_t const &input,
                                       positive_int kernel_size,
                                       positive_int stride) {
  TensorDims dims = cgb.get_shape(input).dims;
  int input_height =
      dim_at_idx(dims, relative_ff_dim_t{2}).int_from_positive_int();
  int input_width =
      dim_at_idx(dims, relative_ff_dim_t{3}).int_from_positive_int();
  int kernel = kernel_size.int_from_positive_int();
  int stride_value = stride.int_from_positive_int();

  auto [top, bottom] =
      get_same_padding_for_dim(input_height, kernel, stride_value);
  auto [left, right] =
      get_same_padding_for_dim(input_width, kernel, stride_value);
  return SpatialPadding{top, bottom, left, right};
}

static tensor_guid_t create_constant_like(ComputationGraphBuilder &cgb,
                                          tensor_guid_t const &input,
                                          float value,
                                          std::string const &name) {
  tensor_guid_t result = cgb.scalar_multiply(input, 0.0f, name + ".zero");
  if (value != 0.0f) {
    result = cgb.scalar_add(result, value, name + ".value");
  }
  return result;
}

static tensor_guid_t
    create_constant_padding_for_axis(ComputationGraphBuilder &cgb,
                                     tensor_guid_t const &input,
                                     relative_ff_dim_t axis,
                                     int padding_before,
                                     int padding_after,
                                     float value,
                                     std::string const &name) {
  if (padding_before == 0 && padding_after == 0) {
    return input;
  }

  int axis_size =
      dim_at_idx(cgb.get_shape(input).dims, axis).int_from_positive_int();
  ASSERT(padding_before >= 0 && padding_after >= 0);
  ASSERT(padding_before + padding_after < axis_size);

  std::vector<positive_int> split_sizes;
  if (padding_before > 0) {
    split_sizes.push_back(positive_int{padding_before});
  }
  split_sizes.push_back(
      positive_int{axis_size - padding_before - padding_after});
  if (padding_after > 0) {
    split_sizes.push_back(positive_int{padding_after});
  }

  std::vector<tensor_guid_t> pieces =
      cgb.split(input, split_sizes, axis, name + ".split");
  std::vector<tensor_guid_t> concat_inputs;
  if (padding_before > 0) {
    concat_inputs.push_back(
        create_constant_like(cgb, pieces.front(), value, name + ".before"));
  }
  concat_inputs.insert(concat_inputs.end(), pieces.begin(), pieces.end());
  if (padding_after > 0) {
    concat_inputs.push_back(
        create_constant_like(cgb, pieces.back(), value, name + ".after"));
  }
  return cgb.concat(concat_inputs, axis, name + ".concat");
}

static tensor_guid_t create_constant_padding2d(ComputationGraphBuilder &cgb,
                                               tensor_guid_t const &input,
                                               SpatialPadding const &padding,
                                               float value,
                                               std::string const &name) {
  tensor_guid_t result = create_constant_padding_for_axis(cgb,
                                                          input,
                                                          relative_ff_dim_t{2},
                                                          padding.top,
                                                          padding.bottom,
                                                          value,
                                                          name + ".height");
  return create_constant_padding_for_axis(cgb,
                                          result,
                                          relative_ff_dim_t{3},
                                          padding.left,
                                          padding.right,
                                          value,
                                          name + ".width");
}

static tensor_guid_t
    create_shifted_reduction_input(ComputationGraphBuilder &cgb,
                                   tensor_guid_t const &input,
                                   std::string const &name) {
  TensorDims dims = cgb.get_shape(input).dims;
  positive_int height = dim_at_idx(dims, relative_ff_dim_t{2});
  positive_int width = dim_at_idx(dims, relative_ff_dim_t{3});
  ASSERT(height > 1_p && width > 1_p);

  std::vector<tensor_guid_t> height_parts =
      cgb.split(input,
                {1_p, positive_int{height.int_from_positive_int() - 1}},
                relative_ff_dim_t{2},
                name + ".crop_top");
  tensor_guid_t zero_row =
      create_constant_like(cgb, height_parts.at(0), 0.0f, name + ".pad_bottom");
  tensor_guid_t shifted_height = cgb.concat({height_parts.at(1), zero_row},
                                            relative_ff_dim_t{2},
                                            name + ".shift_height");

  std::vector<tensor_guid_t> width_parts =
      cgb.split(shifted_height,
                {1_p, positive_int{width.int_from_positive_int() - 1}},
                relative_ff_dim_t{3},
                name + ".crop_left");
  tensor_guid_t zero_column =
      create_constant_like(cgb, width_parts.at(0), 0.0f, name + ".pad_right");
  return cgb.concat({width_parts.at(1), zero_column},
                    relative_ff_dim_t{3},
                    name + ".shift_width");
}

static positive_int get_num_channels(ComputationGraphBuilder const &cgb,
                                     tensor_guid_t const &input) {
  return dim_at_idx(cgb.get_shape(input).dims, relative_ff_dim_t{1});
}

static tensor_guid_t create_same_pool2d(ComputationGraphBuilder &cgb,
                                        tensor_guid_t const &input,
                                        positive_int kernel_size,
                                        positive_int stride,
                                        PoolOp type,
                                        std::string const &name) {
  SpatialPadding padding = get_same_padding(cgb, input, kernel_size, stride);
  bool is_symmetric =
      padding.top == padding.bottom && padding.left == padding.right;
  // timm materializes padding for every stride-two SAME pool. This matters for
  // average pooling because those explicit zeros participate in the divisor;
  // symmetric max-pool padding is equivalent to native padding.
  bool use_native_padding =
      is_symmetric && (stride == 1_p || type == PoolOp::MAX);

  tensor_guid_t padded_input = input;
  nonnegative_int native_padding_h = 0_n;
  nonnegative_int native_padding_w = 0_n;
  if (use_native_padding) {
    native_padding_h = nonnegative_int{padding.top};
    native_padding_w = nonnegative_int{padding.left};
  } else {
    float padding_value =
        type == PoolOp::MAX ? std::numeric_limits<float>::lowest() : 0.0f;
    padded_input = create_constant_padding2d(
        cgb, input, padding, padding_value, name + ".same_pad");
  }

  return cgb.pool2d(padded_input,
                    /*kernelH=*/kernel_size,
                    /*kernelW=*/kernel_size,
                    /*strideH=*/stride,
                    /*strideW=*/stride,
                    /*paddingH=*/native_padding_h,
                    /*paddingW=*/native_padding_w,
                    /*type=*/type,
                    /*activation=*/std::nullopt,
                    /*name=*/name);
}

static tensor_guid_t create_act_conv_bn(ComputationGraphBuilder &cgb,
                                        tensor_guid_t const &input,
                                        positive_int out_channels,
                                        std::string const &name) {
  tensor_guid_t x = cgb.relu(input, name + ".act");
  x = cgb.conv2d(x,
                 /*outChannels=*/out_channels,
                 /*kernelH=*/1_p,
                 /*kernelW=*/1_p,
                 /*strideH=*/1_p,
                 /*strideW=*/1_p,
                 /*paddingH=*/0_n,
                 /*paddingW=*/0_n,
                 /*activation=*/std::nullopt,
                 /*groups=*/1_p,
                 /*use_bias=*/false,
                 /*kernel_initializer=*/std::nullopt,
                 /*bias_initializer=*/std::nullopt,
                 /*kernel_regularizer=*/std::nullopt,
                 /*name=*/name + ".conv");
  return cgb.batch_norm(x,
                        /*affine=*/true,
                        /*activation=*/std::nullopt,
                        /*eps=*/0.001f,
                        /*momentum=*/0.1f,
                        /*name=*/name + ".bn");
}

static tensor_guid_t create_separable_conv2d(ComputationGraphBuilder &cgb,
                                             tensor_guid_t const &input,
                                             positive_int out_channels,
                                             positive_int kernel_size,
                                             positive_int stride,
                                             std::string const &name) {
  positive_int in_channels = get_num_channels(cgb, input);
  SpatialPadding padding = get_same_padding(cgb, input, kernel_size, stride);
  bool is_symmetric =
      padding.top == padding.bottom && padding.left == padding.right;

  tensor_guid_t padded_input = input;
  nonnegative_int native_padding_h = 0_n;
  nonnegative_int native_padding_w = 0_n;
  if (is_symmetric) {
    native_padding_h = nonnegative_int{padding.top};
    native_padding_w = nonnegative_int{padding.left};
  } else {
    padded_input = create_constant_padding2d(
        cgb, input, padding, 0.0f, name + ".same_pad");
  }

  tensor_guid_t x = cgb.conv2d(padded_input,
                               /*outChannels=*/in_channels,
                               /*kernelH=*/kernel_size,
                               /*kernelW=*/kernel_size,
                               /*strideH=*/stride,
                               /*strideW=*/stride,
                               /*paddingH=*/native_padding_h,
                               /*paddingW=*/native_padding_w,
                               /*activation=*/std::nullopt,
                               /*groups=*/in_channels,
                               /*use_bias=*/false,
                               /*kernel_initializer=*/std::nullopt,
                               /*bias_initializer=*/std::nullopt,
                               /*kernel_regularizer=*/std::nullopt,
                               /*name=*/name + ".depthwise");
  return cgb.conv2d(x,
                    /*outChannels=*/out_channels,
                    /*kernelH=*/1_p,
                    /*kernelW=*/1_p,
                    /*strideH=*/1_p,
                    /*strideW=*/1_p,
                    /*paddingH=*/0_n,
                    /*paddingW=*/0_n,
                    /*activation=*/std::nullopt,
                    /*groups=*/1_p,
                    /*use_bias=*/false,
                    /*kernel_initializer=*/std::nullopt,
                    /*bias_initializer=*/std::nullopt,
                    /*kernel_regularizer=*/std::nullopt,
                    /*name=*/name + ".pointwise");
}

static tensor_guid_t create_branch_separables(ComputationGraphBuilder &cgb,
                                              tensor_guid_t const &input,
                                              positive_int out_channels,
                                              positive_int kernel_size,
                                              positive_int stride,
                                              bool stem_cell,
                                              std::string const &name) {
  positive_int middle_channels =
      stem_cell ? out_channels : get_num_channels(cgb, input);

  tensor_guid_t x = cgb.relu(input, name + ".act_1");
  x = create_separable_conv2d(
      cgb, x, middle_channels, kernel_size, stride, name + ".separable_1");
  x = cgb.batch_norm(x,
                     /*affine=*/true,
                     /*activation=*/std::nullopt,
                     /*eps=*/0.001f,
                     /*momentum=*/0.1f,
                     /*name=*/name + ".bn_sep_1");
  x = cgb.relu(x, name + ".act_2");
  x = create_separable_conv2d(cgb,
                              x,
                              out_channels,
                              kernel_size,
                              /*stride=*/1_p,
                              name + ".separable_2");
  return cgb.batch_norm(x,
                        /*affine=*/true,
                        /*activation=*/std::nullopt,
                        /*eps=*/0.001f,
                        /*momentum=*/0.1f,
                        /*name=*/name + ".bn_sep_2");
}

static tensor_guid_t
    create_factorized_reduction(ComputationGraphBuilder &cgb,
                                tensor_guid_t const &input,
                                positive_int out_channels_per_path,
                                std::string const &name) {
  tensor_guid_t activated = cgb.relu(input, name + ".act");

  tensor_guid_t path_1 = cgb.pool2d(activated,
                                    /*kernelH=*/1_p,
                                    /*kernelW=*/1_p,
                                    /*strideH=*/2_p,
                                    /*strideW=*/2_p,
                                    /*paddingH=*/0_n,
                                    /*paddingW=*/0_n,
                                    /*type=*/PoolOp::AVG,
                                    /*activation=*/std::nullopt,
                                    /*name=*/name + ".path_1.avgpool");
  path_1 = cgb.conv2d(path_1,
                      /*outChannels=*/out_channels_per_path,
                      /*kernelH=*/1_p,
                      /*kernelW=*/1_p,
                      /*strideH=*/1_p,
                      /*strideW=*/1_p,
                      /*paddingH=*/0_n,
                      /*paddingW=*/0_n,
                      /*activation=*/std::nullopt,
                      /*groups=*/1_p,
                      /*use_bias=*/false,
                      /*kernel_initializer=*/std::nullopt,
                      /*bias_initializer=*/std::nullopt,
                      /*kernel_regularizer=*/std::nullopt,
                      /*name=*/name + ".path_1.conv");

  tensor_guid_t shifted =
      create_shifted_reduction_input(cgb, activated, name + ".path_2.shift");
  tensor_guid_t path_2 = cgb.pool2d(shifted,
                                    /*kernelH=*/1_p,
                                    /*kernelW=*/1_p,
                                    /*strideH=*/2_p,
                                    /*strideW=*/2_p,
                                    /*paddingH=*/0_n,
                                    /*paddingW=*/0_n,
                                    /*type=*/PoolOp::AVG,
                                    /*activation=*/std::nullopt,
                                    /*name=*/name + ".path_2.avgpool");
  path_2 = cgb.conv2d(path_2,
                      /*outChannels=*/out_channels_per_path,
                      /*kernelH=*/1_p,
                      /*kernelW=*/1_p,
                      /*strideH=*/1_p,
                      /*strideW=*/1_p,
                      /*paddingH=*/0_n,
                      /*paddingW=*/0_n,
                      /*activation=*/std::nullopt,
                      /*groups=*/1_p,
                      /*use_bias=*/false,
                      /*kernel_initializer=*/std::nullopt,
                      /*bias_initializer=*/std::nullopt,
                      /*kernel_regularizer=*/std::nullopt,
                      /*name=*/name + ".path_2.conv");

  tensor_guid_t output = cgb.concat({path_1, path_2},
                                    /*axis=*/relative_ff_dim_t{1},
                                    /*name=*/name + ".concat");
  return cgb.batch_norm(output,
                        /*affine=*/true,
                        /*activation=*/std::nullopt,
                        /*eps=*/0.001f,
                        /*momentum=*/0.1f,
                        /*name=*/name + ".final_path_bn");
}

static tensor_guid_t create_nasnet_stem_cell_0(ComputationGraphBuilder &cgb,
                                               tensor_guid_t const &input,
                                               positive_int channels,
                                               std::string const &name) {
  tensor_guid_t x1 =
      create_act_conv_bn(cgb, input, channels, name + ".conv_1x1");

  tensor_guid_t iter_0 =
      cgb.add(create_branch_separables(cgb,
                                       x1,
                                       channels,
                                       /*kernel_size=*/5_p,
                                       /*stride=*/2_p,
                                       /*stem_cell=*/false,
                                       name + ".comb_iter_0_left"),
              create_branch_separables(cgb,
                                       input,
                                       channels,
                                       /*kernel_size=*/7_p,
                                       /*stride=*/2_p,
                                       /*stem_cell=*/true,
                                       name + ".comb_iter_0_right"),
              name + ".comb_iter_0");

  tensor_guid_t iter_1 =
      cgb.add(create_same_pool2d(cgb,
                                 x1,
                                 /*kernel_size=*/3_p,
                                 /*stride=*/2_p,
                                 PoolOp::MAX,
                                 name + ".comb_iter_1_left"),
              create_branch_separables(cgb,
                                       input,
                                       channels,
                                       /*kernel_size=*/7_p,
                                       /*stride=*/2_p,
                                       /*stem_cell=*/true,
                                       name + ".comb_iter_1_right"),
              name + ".comb_iter_1");

  tensor_guid_t iter_2 =
      cgb.add(create_same_pool2d(cgb,
                                 x1,
                                 /*kernel_size=*/3_p,
                                 /*stride=*/2_p,
                                 PoolOp::AVG,
                                 name + ".comb_iter_2_left"),
              create_branch_separables(cgb,
                                       input,
                                       channels,
                                       /*kernel_size=*/5_p,
                                       /*stride=*/2_p,
                                       /*stem_cell=*/true,
                                       name + ".comb_iter_2_right"),
              name + ".comb_iter_2");

  tensor_guid_t iter_3 =
      cgb.add(create_same_pool2d(cgb,
                                 iter_0,
                                 /*kernel_size=*/3_p,
                                 /*stride=*/1_p,
                                 PoolOp::AVG,
                                 name + ".comb_iter_3_right"),
              iter_1,
              name + ".comb_iter_3");

  tensor_guid_t iter_4 =
      cgb.add(create_branch_separables(cgb,
                                       iter_0,
                                       channels,
                                       /*kernel_size=*/3_p,
                                       /*stride=*/1_p,
                                       /*stem_cell=*/false,
                                       name + ".comb_iter_4_left"),
              create_same_pool2d(cgb,
                                 x1,
                                 /*kernel_size=*/3_p,
                                 /*stride=*/2_p,
                                 PoolOp::MAX,
                                 name + ".comb_iter_4_right"),
              name + ".comb_iter_4");

  return cgb.concat({iter_1, iter_2, iter_3, iter_4},
                    /*axis=*/relative_ff_dim_t{1},
                    /*name=*/name + ".output");
}

static tensor_guid_t create_nasnet_stem_cell_1(ComputationGraphBuilder &cgb,
                                               tensor_guid_t const &x_conv0,
                                               tensor_guid_t const &x_stem_0,
                                               positive_int channels,
                                               std::string const &name) {
  tensor_guid_t x_left =
      create_act_conv_bn(cgb, x_stem_0, channels, name + ".conv_1x1");
  tensor_guid_t x_right = create_factorized_reduction(
      cgb,
      x_conv0,
      positive_int{channels.int_from_positive_int() / 2},
      name + ".previous_reduction");

  tensor_guid_t iter_0 =
      cgb.add(create_branch_separables(cgb,
                                       x_left,
                                       channels,
                                       /*kernel_size=*/5_p,
                                       /*stride=*/2_p,
                                       /*stem_cell=*/false,
                                       name + ".comb_iter_0_left"),
              create_branch_separables(cgb,
                                       x_right,
                                       channels,
                                       /*kernel_size=*/7_p,
                                       /*stride=*/2_p,
                                       /*stem_cell=*/false,
                                       name + ".comb_iter_0_right"),
              name + ".comb_iter_0");

  tensor_guid_t iter_1 =
      cgb.add(create_same_pool2d(cgb,
                                 x_left,
                                 /*kernel_size=*/3_p,
                                 /*stride=*/2_p,
                                 PoolOp::MAX,
                                 name + ".comb_iter_1_left"),
              create_branch_separables(cgb,
                                       x_right,
                                       channels,
                                       /*kernel_size=*/7_p,
                                       /*stride=*/2_p,
                                       /*stem_cell=*/false,
                                       name + ".comb_iter_1_right"),
              name + ".comb_iter_1");

  tensor_guid_t iter_2 =
      cgb.add(create_same_pool2d(cgb,
                                 x_left,
                                 /*kernel_size=*/3_p,
                                 /*stride=*/2_p,
                                 PoolOp::AVG,
                                 name + ".comb_iter_2_left"),
              create_branch_separables(cgb,
                                       x_right,
                                       channels,
                                       /*kernel_size=*/5_p,
                                       /*stride=*/2_p,
                                       /*stem_cell=*/false,
                                       name + ".comb_iter_2_right"),
              name + ".comb_iter_2");

  tensor_guid_t iter_3 =
      cgb.add(create_same_pool2d(cgb,
                                 iter_0,
                                 /*kernel_size=*/3_p,
                                 /*stride=*/1_p,
                                 PoolOp::AVG,
                                 name + ".comb_iter_3_right"),
              iter_1,
              name + ".comb_iter_3");

  tensor_guid_t iter_4 =
      cgb.add(create_branch_separables(cgb,
                                       iter_0,
                                       channels,
                                       /*kernel_size=*/3_p,
                                       /*stride=*/1_p,
                                       /*stem_cell=*/false,
                                       name + ".comb_iter_4_left"),
              create_same_pool2d(cgb,
                                 x_left,
                                 /*kernel_size=*/3_p,
                                 /*stride=*/2_p,
                                 PoolOp::MAX,
                                 name + ".comb_iter_4_right"),
              name + ".comb_iter_4");

  return cgb.concat({iter_1, iter_2, iter_3, iter_4},
                    /*axis=*/relative_ff_dim_t{1},
                    /*name=*/name + ".output");
}

static tensor_guid_t
    create_nasnet_first_cell(ComputationGraphBuilder &cgb,
                             tensor_guid_t const &x,
                             tensor_guid_t const &x_prev,
                             positive_int out_channels_per_left_path,
                             positive_int out_channels,
                             std::string const &name) {
  tensor_guid_t x_left = create_factorized_reduction(
      cgb, x_prev, out_channels_per_left_path, name + ".previous_reduction");
  tensor_guid_t x_right =
      create_act_conv_bn(cgb, x, out_channels, name + ".conv_1x1");

  tensor_guid_t iter_0 =
      cgb.add(create_branch_separables(cgb,
                                       x_right,
                                       out_channels,
                                       5_p,
                                       1_p,
                                       false,
                                       name + ".comb_iter_0_left"),
              create_branch_separables(cgb,
                                       x_left,
                                       out_channels,
                                       3_p,
                                       1_p,
                                       false,
                                       name + ".comb_iter_0_right"),
              name + ".comb_iter_0");
  tensor_guid_t iter_1 =
      cgb.add(create_branch_separables(cgb,
                                       x_left,
                                       out_channels,
                                       5_p,
                                       1_p,
                                       false,
                                       name + ".comb_iter_1_left"),
              create_branch_separables(cgb,
                                       x_left,
                                       out_channels,
                                       3_p,
                                       1_p,
                                       false,
                                       name + ".comb_iter_1_right"),
              name + ".comb_iter_1");
  tensor_guid_t iter_2 = cgb.add(
      create_same_pool2d(
          cgb, x_right, 3_p, 1_p, PoolOp::AVG, name + ".comb_iter_2_left"),
      x_left,
      name + ".comb_iter_2");
  tensor_guid_t iter_3 = cgb.add(
      create_same_pool2d(
          cgb, x_left, 3_p, 1_p, PoolOp::AVG, name + ".comb_iter_3_left"),
      create_same_pool2d(
          cgb, x_left, 3_p, 1_p, PoolOp::AVG, name + ".comb_iter_3_right"),
      name + ".comb_iter_3");
  tensor_guid_t iter_4 =
      cgb.add(create_branch_separables(cgb,
                                       x_right,
                                       out_channels,
                                       3_p,
                                       1_p,
                                       false,
                                       name + ".comb_iter_4_left"),
              x_right,
              name + ".comb_iter_4");

  return cgb.concat({x_left, iter_0, iter_1, iter_2, iter_3, iter_4},
                    relative_ff_dim_t{1},
                    name + ".output");
}

static tensor_guid_t create_nasnet_normal_cell(ComputationGraphBuilder &cgb,
                                               tensor_guid_t const &x,
                                               tensor_guid_t const &x_prev,
                                               positive_int channels,
                                               std::string const &name) {
  tensor_guid_t x_left =
      create_act_conv_bn(cgb, x_prev, channels, name + ".conv_prev_1x1");
  tensor_guid_t x_right =
      create_act_conv_bn(cgb, x, channels, name + ".conv_1x1");

  tensor_guid_t iter_0 = cgb.add(
      create_branch_separables(
          cgb, x_right, channels, 5_p, 1_p, false, name + ".comb_iter_0_left"),
      create_branch_separables(
          cgb, x_left, channels, 3_p, 1_p, false, name + ".comb_iter_0_right"),
      name + ".comb_iter_0");
  tensor_guid_t iter_1 = cgb.add(
      create_branch_separables(
          cgb, x_left, channels, 5_p, 1_p, false, name + ".comb_iter_1_left"),
      create_branch_separables(
          cgb, x_left, channels, 3_p, 1_p, false, name + ".comb_iter_1_right"),
      name + ".comb_iter_1");
  tensor_guid_t iter_2 = cgb.add(
      create_same_pool2d(
          cgb, x_right, 3_p, 1_p, PoolOp::AVG, name + ".comb_iter_2_left"),
      x_left,
      name + ".comb_iter_2");
  tensor_guid_t iter_3 = cgb.add(
      create_same_pool2d(
          cgb, x_left, 3_p, 1_p, PoolOp::AVG, name + ".comb_iter_3_left"),
      create_same_pool2d(
          cgb, x_left, 3_p, 1_p, PoolOp::AVG, name + ".comb_iter_3_right"),
      name + ".comb_iter_3");
  tensor_guid_t iter_4 = cgb.add(
      create_branch_separables(
          cgb, x_right, channels, 3_p, 1_p, false, name + ".comb_iter_4_left"),
      x_right,
      name + ".comb_iter_4");

  return cgb.concat({x_left, iter_0, iter_1, iter_2, iter_3, iter_4},
                    relative_ff_dim_t{1},
                    name + ".output");
}

static tensor_guid_t create_nasnet_reduction_cell(ComputationGraphBuilder &cgb,
                                                  tensor_guid_t const &x,
                                                  tensor_guid_t const &x_prev,
                                                  positive_int channels,
                                                  std::string const &name) {
  tensor_guid_t x_left =
      create_act_conv_bn(cgb, x_prev, channels, name + ".conv_prev_1x1");
  tensor_guid_t x_right =
      create_act_conv_bn(cgb, x, channels, name + ".conv_1x1");

  tensor_guid_t iter_0 = cgb.add(
      create_branch_separables(
          cgb, x_right, channels, 5_p, 2_p, false, name + ".comb_iter_0_left"),
      create_branch_separables(
          cgb, x_left, channels, 7_p, 2_p, false, name + ".comb_iter_0_right"),
      name + ".comb_iter_0");
  tensor_guid_t iter_1 = cgb.add(
      create_same_pool2d(
          cgb, x_right, 3_p, 2_p, PoolOp::MAX, name + ".comb_iter_1_left"),
      create_branch_separables(
          cgb, x_left, channels, 7_p, 2_p, false, name + ".comb_iter_1_right"),
      name + ".comb_iter_1");
  tensor_guid_t iter_2 = cgb.add(
      create_same_pool2d(
          cgb, x_right, 3_p, 2_p, PoolOp::AVG, name + ".comb_iter_2_left"),
      create_branch_separables(
          cgb, x_left, channels, 5_p, 2_p, false, name + ".comb_iter_2_right"),
      name + ".comb_iter_2");
  tensor_guid_t iter_3 = cgb.add(
      create_same_pool2d(
          cgb, iter_0, 3_p, 1_p, PoolOp::AVG, name + ".comb_iter_3_right"),
      iter_1,
      name + ".comb_iter_3");
  tensor_guid_t iter_4 = cgb.add(
      create_branch_separables(
          cgb, iter_0, channels, 3_p, 1_p, false, name + ".comb_iter_4_left"),
      create_same_pool2d(
          cgb, x_right, 3_p, 2_p, PoolOp::MAX, name + ".comb_iter_4_right"),
      name + ".comb_iter_4");

  return cgb.concat(
      {iter_1, iter_2, iter_3, iter_4}, relative_ff_dim_t{1}, name + ".output");
}

} // namespace

NASNetALargeConfig get_default_nasnet_a_large_config() {
  return NASNetALargeConfig{
      /*batch_size=*/32_p,
      /*num_classes=*/1000_p,
      /*input_channels=*/3_p,
      /*image_height=*/331_p,
      /*image_width=*/331_p,
      /*dropout=*/0.0f,
  };
}

ComputationGraph
    get_nasnet_a_large_computation_graph(NASNetALargeConfig const &config) {
  if (!(config.dropout >= 0.0f && config.dropout < 1.0f)) {
    throw mk_runtime_error(fmt::format(
        "NASNet-A Large dropout must be in the range [0, 1), but found {}",
        config.dropout));
  }

  ComputationGraphBuilder cgb;

  TensorShape input_shape = TensorShape{
      TensorDims{FFOrdered<positive_int>{config.batch_size,
                                         config.input_channels,
                                         config.image_height,
                                         config.image_width}},
      DataType::FLOAT,
  };
  tensor_guid_t input = cgb.create_input(input_shape, CreateGrad::YES, "input");

  tensor_guid_t x_conv0 = cgb.conv2d(input,
                                     /*outChannels=*/96_p,
                                     /*kernelH=*/3_p,
                                     /*kernelW=*/3_p,
                                     /*strideH=*/2_p,
                                     /*strideW=*/2_p,
                                     /*paddingH=*/0_n,
                                     /*paddingW=*/0_n,
                                     /*activation=*/std::nullopt,
                                     /*groups=*/1_p,
                                     /*use_bias=*/false,
                                     /*kernel_initializer=*/std::nullopt,
                                     /*bias_initializer=*/std::nullopt,
                                     /*kernel_regularizer=*/std::nullopt,
                                     /*name=*/"conv0.conv");
  x_conv0 = cgb.batch_norm(x_conv0,
                           /*affine=*/true,
                           /*activation=*/std::nullopt,
                           /*eps=*/0.001f,
                           /*momentum=*/0.1f,
                           /*name=*/"conv0.bn");

  tensor_guid_t x_stem_0 =
      create_nasnet_stem_cell_0(cgb, x_conv0, 42_p, "cell_stem_0");
  tensor_guid_t x_stem_1 =
      create_nasnet_stem_cell_1(cgb, x_conv0, x_stem_0, 84_p, "cell_stem_1");

  tensor_guid_t previous = x_stem_0;
  tensor_guid_t current =
      create_nasnet_first_cell(cgb, x_stem_1, previous, 84_p, 168_p, "cell_0");
  previous = x_stem_1;

  for (int idx = 1; idx <= 5; idx++) {
    tensor_guid_t next = create_nasnet_normal_cell(
        cgb, current, previous, 168_p, fmt::format("cell_{}", idx));
    previous = current;
    current = next;
  }

  tensor_guid_t reduction_0 = create_nasnet_reduction_cell(
      cgb, current, previous, 336_p, "reduction_cell_0");
  current = create_nasnet_first_cell(
      cgb, reduction_0, previous, 168_p, 336_p, "cell_6");
  previous = reduction_0;

  for (int idx = 7; idx <= 11; idx++) {
    tensor_guid_t next = create_nasnet_normal_cell(
        cgb, current, previous, 336_p, fmt::format("cell_{}", idx));
    previous = current;
    current = next;
  }

  tensor_guid_t reduction_1 = create_nasnet_reduction_cell(
      cgb, current, previous, 672_p, "reduction_cell_1");
  current = create_nasnet_first_cell(
      cgb, reduction_1, previous, 336_p, 672_p, "cell_12");
  previous = reduction_1;

  for (int idx = 13; idx <= 17; idx++) {
    tensor_guid_t next = create_nasnet_normal_cell(
        cgb, current, previous, 672_p, fmt::format("cell_{}", idx));
    previous = current;
    current = next;
  }

  tensor_guid_t output = cgb.relu(current, "act");
  output = cgb.adaptive_pool2d(output,
                               /*output_h=*/1_p,
                               /*output_w=*/1_p,
                               /*type=*/PoolOp::AVG,
                               /*activation=*/std::nullopt,
                               /*name=*/"global_pool");
  if (config.dropout > 0.0f) {
    output = cgb.dropout(output,
                         /*rate=*/config.dropout,
                         /*seed=*/0,
                         /*name=*/"head_drop");
  }
  output = cgb.flat(output,
                    /*start_dim=*/relative_ff_dim_t{1},
                    /*end_dim=*/std::nullopt,
                    /*name=*/"flatten");
  output = cgb.dense(output,
                     /*outDim=*/config.num_classes,
                     /*activation=*/std::nullopt,
                     /*use_bias=*/true,
                     /*data_type=*/DataType::FLOAT,
                     /*projection_initializer=*/std::nullopt,
                     /*bias_initializer=*/std::nullopt,
                     /*name=*/"last_linear");

  return cgb.computation_graph;
}

} // namespace FlexFlow
