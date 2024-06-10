#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_layer_attrs.h"
#include "test/utils/doctest.h"
#include "utils/containers.h"
#include "utils/containers/without_nullopts.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("ParallelComputationGraphBuilder") {
    ParallelComputationGraphBuilder b;

    size_t batch_size = 2;

    TensorShape unpar_input_shape = TensorShape{
        TensorDims{FFOrdered<size_t>{batch_size, 3, 10, 10}},
        DataType::FLOAT,
    };

    ParallelTensorShape input_shape =
        lift_to_parallel_with_degrees(unpar_input_shape,
                                      SumDegree{1},
                                      DiscardCopyDegree{1},
                                      FFOrdered<int>{2, 1, 1, 1});

    parallel_tensor_guid_t input = b.create_input_tensor(input_shape);

    int outChannels = 6;
    int kernelH = 5;
    int kernelW = 4;
    int strideH = 3;
    int strideW = 2;
    int paddingH = 1;
    int paddingW = 0;
    parallel_tensor_guid_t output = b.conv2d(input,
                                             /*outChannels=*/outChannels,
                                             /*kernelH=*/kernelH,
                                             /*kernelW=*/kernelW,
                                             /*strideH=*/strideH,
                                             /*strideW=*/strideW,
                                             /*paddingH=*/paddingH,
                                             /*paddingW=*/paddingW);

    std::unordered_map<parallel_layer_guid_t, ParallelLayerAttrs> layers =
        generate_map(get_parallel_layers(b.pcg),
                     [&](parallel_layer_guid_t const &l) {
                       return get_parallel_layer_attrs(b.pcg, l);
                     });
    CHECK_MESSAGE(layers.size() == 4, "Incorrect layers ", layers);

    auto num_attrs_of_type = [&](OperatorType op_type) -> int {
      return count(values(layers), [&](ParallelLayerAttrs const &l) {
        return get_op_type(l) == op_type;
      });
    };

    int num_weight_attrs = num_attrs_of_type(OperatorType::WEIGHT);
    CHECK(num_weight_attrs == 2);

    int num_input_attrs = num_attrs_of_type(OperatorType::INPUT);
    CHECK(num_input_attrs == 1);

    int num_conv_attrs = num_attrs_of_type(OperatorType::CONV2D);
    CHECK(num_conv_attrs == 1);

    parallel_layer_guid_t conv_guid = get_only(without_nullopts(transform(
        as_vector(items(layers)),
        [](std::pair<parallel_layer_guid_t, ParallelLayerAttrs> const &kv)
            -> std::optional<parallel_layer_guid_t> {
          if (get_op_type(kv.second) == OperatorType::CONV2D) {
            return kv.first;
          } else {
            return std::nullopt;
          }
        })));
    Conv2DAttrs conv_attrs = layers.at(conv_guid).op_attrs.get<Conv2DAttrs>();
    Conv2DAttrs correct_attrs = Conv2DAttrs{
        outChannels,
        kernelH,
        kernelW,
        strideH,
        strideW,
        paddingH,
        paddingW,
        /*groups=*/1,
        /*activation=*/std::nullopt,
        /*use_bias=*/true,
    };
    CHECK(conv_attrs == correct_attrs);

    ParallelTensorShape correct_output_shape =
        get_output_shape(correct_attrs, input_shape);
    ParallelTensorShape correct_kernel_shape =
        get_kernel_shape(correct_attrs, input_shape);
    ParallelTensorShape correct_bias_shape =
        get_bias_shape(correct_attrs, input_shape);

    std::vector<parallel_tensor_guid_t> conv_inputs =
        get_layer_inputs(b.pcg, conv_guid);

    parallel_tensor_guid_t conv_input = conv_inputs.at(0);
    ParallelTensorShape conv_input_shape =
        get_parallel_tensor_attrs(b.pcg, conv_input).shape;
    CHECK(conv_input_shape == input_shape);

    parallel_tensor_guid_t conv_kernel = conv_inputs.at(1);
    ParallelTensorShape conv_kernel_shape =
        get_parallel_tensor_attrs(b.pcg, conv_kernel).shape;
    CHECK(conv_kernel_shape == correct_kernel_shape);

    parallel_tensor_guid_t conv_bias = conv_inputs.at(2);
    ParallelTensorShape conv_bias_shape =
        get_parallel_tensor_attrs(b.pcg, conv_bias).shape;
    CHECK(conv_bias_shape == correct_bias_shape);

    std::vector<parallel_tensor_guid_t> conv_outputs =
        get_layer_outputs(b.pcg, conv_guid);
    CHECK(conv_outputs.size() == 1);

    parallel_tensor_guid_t conv_output = get_only(conv_outputs);
    ParallelTensorShape conv_output_shape =
        get_parallel_tensor_attrs(b.pcg, conv_output).shape;
    CHECK(conv_output_shape == correct_output_shape);
  };
}
