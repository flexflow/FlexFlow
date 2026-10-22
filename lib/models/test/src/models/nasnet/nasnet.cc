#include "models/nasnet/nasnet.h"
#include "op-attrs/ops/conv_2d_attrs.dtg.h"
#include "op-attrs/tensor_shape.h"
#include "pcg/computation_graph.h"

#include <doctest/doctest.h>
#include <map>
#include <set>
#include <string>

using namespace ::FlexFlow;

namespace {

TensorShape get_named_layer_output_shape(ComputationGraph const &graph,
                                         std::string const &name) {
  layer_guid_t layer = get_layer_by_name(graph, name);
  tensor_guid_t output =
      get_outgoing_tensors(graph, layer).at(TensorSlotName::OUTPUT);
  return get_tensor_attrs(graph, output).shape;
}

TensorShape make_4d_shape(positive_int batch_size,
                          positive_int channels,
                          positive_int height,
                          positive_int width) {
  return TensorShape{
      TensorDims{FFOrdered<positive_int>{batch_size, channels, height, width}},
      DataType::FLOAT,
  };
}

} // namespace

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_default_nasnet_a_large_config") {
    NASNetALargeConfig config = get_default_nasnet_a_large_config();

    CHECK(config.batch_size == 32_p);
    CHECK(config.num_classes == 1000_p);
    CHECK(config.input_channels == 3_p);
    CHECK(config.image_height == 331_p);
    CHECK(config.image_width == 331_p);
    CHECK(config.dropout == 0.0f);
  }

  TEST_CASE("get_nasnet_a_large_computation_graph") {
    NASNetALargeConfig config = get_default_nasnet_a_large_config();
    ComputationGraph result = get_nasnet_a_large_computation_graph(config);

    std::map<std::string, TensorShape> expected = {
        {"conv0.bn", make_4d_shape(32_p, 96_p, 165_p, 165_p)},
        {"cell_stem_0.output", make_4d_shape(32_p, 168_p, 83_p, 83_p)},
        {"cell_stem_1.output", make_4d_shape(32_p, 336_p, 42_p, 42_p)},
        {"cell_5.output", make_4d_shape(32_p, 1008_p, 42_p, 42_p)},
        {"reduction_cell_0.output", make_4d_shape(32_p, 1344_p, 21_p, 21_p)},
        {"cell_11.output", make_4d_shape(32_p, 2016_p, 21_p, 21_p)},
        {"reduction_cell_1.output", make_4d_shape(32_p, 2688_p, 11_p, 11_p)},
        {"cell_17.output", make_4d_shape(32_p, 4032_p, 11_p, 11_p)},
        {"global_pool", make_4d_shape(32_p, 4032_p, 1_p, 1_p)},
    };

    for (auto const &[name, shape] : expected) {
      CHECK(get_named_layer_output_shape(result, name) == shape);
    }

    std::map<OperatorType, positive_int> expected_operator_counts = {
        {OperatorType::INPUT, 1_p},
        {OperatorType::WEIGHT, 1018_p},
        {OperatorType::CONV2D, 488_p},
        {OperatorType::LINEAR, 1_p},
        {OperatorType::POOL2D, 79_p},
        {OperatorType::RELU, 264_p},
        {OperatorType::FLAT, 1_p},
        {OperatorType::BATCHNORM, 264_p},
        {OperatorType::CONCAT, 54_p},
        {OperatorType::EW_ADD, 110_p},
        {OperatorType::SCALAR_ADD, 4_p},
        {OperatorType::SCALAR_MULTIPLY, 42_p},
        {OperatorType::SPLIT, 28_p},
    };
    CHECK(operator_type_counts_in_computation_graph(result) ==
          expected_operator_counts);

    layer_guid_t depthwise_layer = get_layer_by_name(
        result, "cell_17.comb_iter_0_left.separable_1.depthwise");
    Conv2DAttrs depthwise_attrs =
        get_layer_attrs(result, depthwise_layer).op_attrs.require_conv2d();
    CHECK(depthwise_attrs.groups == 672_p);
    CHECK(depthwise_attrs.out_channels == 672_p);
    CHECK(depthwise_attrs.kernel_h == 5_p);
    CHECK(depthwise_attrs.kernel_w == 5_p);

    std::set<tensor_guid_t> unused_tensors = cg_get_unused_tensors(result);
    REQUIRE(unused_tensors.size() == 1);

    TensorShape expected_output = TensorShape{
        TensorDims{FFOrdered<positive_int>{32_p, 1000_p}},
        DataType::FLOAT,
    };
    CHECK(get_tensor_attrs(result, *unused_tensors.begin()).shape ==
          expected_output);
  }

  TEST_CASE("get_nasnet_a_large_computation_graph honors custom config") {
    NASNetALargeConfig config = get_default_nasnet_a_large_config();
    config.batch_size = 2_p;
    config.num_classes = 17_p;
    config.input_channels = 1_p;
    config.image_height = 299_p;
    config.image_width = 299_p;
    config.dropout = 0.25f;

    ComputationGraph result = get_nasnet_a_large_computation_graph(config);

    CHECK(get_named_layer_output_shape(result, "input") ==
          make_4d_shape(2_p, 1_p, 299_p, 299_p));
    CHECK(get_named_layer_output_shape(result, "head_drop") ==
          make_4d_shape(2_p, 4032_p, 1_p, 1_p));

    std::set<tensor_guid_t> unused_tensors = cg_get_unused_tensors(result);
    REQUIRE(unused_tensors.size() == 1);
    TensorShape expected_output = TensorShape{
        TensorDims{FFOrdered<positive_int>{2_p, 17_p}},
        DataType::FLOAT,
    };
    CHECK(get_tensor_attrs(result, *unused_tensors.begin()).shape ==
          expected_output);

    std::map<OperatorType, positive_int> counts =
        operator_type_counts_in_computation_graph(result);
    CHECK(counts.at(OperatorType::DROPOUT) == 1_p);
  }

  TEST_CASE("get_nasnet_a_large_computation_graph rejects invalid dropout") {
    NASNetALargeConfig config = get_default_nasnet_a_large_config();

    config.dropout = -0.1f;
    CHECK_THROWS(get_nasnet_a_large_computation_graph(config));

    config.dropout = 1.0f;
    CHECK_THROWS(get_nasnet_a_large_computation_graph(config));
  }
}
