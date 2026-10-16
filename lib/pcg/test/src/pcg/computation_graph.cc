#include "pcg/computation_graph.h"
#include "op-attrs/ops/linear.h"
#include "pcg/computation_graph_builder.h"
#include "utils/containers/require_only_key.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_incoming_inputs(ComputationGraph, layer_guid_t)") {
    SUBCASE("layer has no inputs") {
      std::string input_name = "input";
      ComputationGraph cg = [&] {
        ComputationGraphBuilder b;

        TensorShape input_shape = TensorShape{
            TensorDims{FFOrdered{
                10_p,
                12_p,
            }},
            DataType::FLOAT,
        };

        tensor_guid_t input =
            b.create_input(input_shape, CreateGrad::YES, input_name);

        return b.computation_graph;
      }();

      layer_guid_t input_layer = get_layer_by_name(cg, input_name);

      std::map<TensorSlotName, tensor_guid_t> result =
          get_incoming_inputs(cg, input_layer);
      std::map<TensorSlotName, tensor_guid_t> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("layer has inputs but no weights") {
      std::string layer_name = "my op";

      ComputationGraphBuilder b;

      TensorShape input_shape = TensorShape{
          TensorDims{FFOrdered{
              10_p,
              12_p,
          }},
          DataType::FLOAT,
      };

      tensor_guid_t input = b.create_input(input_shape, CreateGrad::YES);
      b.relu(input, layer_name);

      ComputationGraph cg = b.computation_graph;

      layer_guid_t layer = get_layer_by_name(cg, layer_name);

      std::map<TensorSlotName, tensor_guid_t> result =
          get_incoming_inputs(cg, layer);
      std::map<TensorSlotName, tensor_guid_t> correct = {
          {
              TensorSlotName::INPUT,
              input,
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("layer has inputs and weights") {
      std::string layer_name = "my op";

      ComputationGraphBuilder b;

      TensorShape input_shape = TensorShape{
          TensorDims{FFOrdered{
              10_p,
              12_p,
          }},
          DataType::FLOAT,
      };

      tensor_guid_t input = b.create_input(input_shape, CreateGrad::YES);
      b.dense(input,
              /*outDim=*/14_p,
              /*activation=*/Activation::RELU,
              /*use_bias=*/true,
              /*data_type=*/DataType::FLOAT,
              /*projection_initializer=*/std::nullopt,
              /*bias_initializer=*/std::nullopt,
              /*name=*/layer_name);

      ComputationGraph cg = b.computation_graph;

      layer_guid_t dense_layer = get_layer_by_name(cg, layer_name);

      std::map<TensorSlotName, tensor_guid_t> result =
          get_incoming_inputs(cg, dense_layer);
      std::map<TensorSlotName, tensor_guid_t> correct = {
          {
              TensorSlotName::INPUT,
              input,
          },
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("get_incoming_weights(ComputationGraph, layer_guid_t)") {
    SUBCASE("layer has no inputs or weights") {
      std::string input_name = "input";
      ComputationGraph cg = [&] {
        ComputationGraphBuilder b;

        TensorShape input_shape = TensorShape{
            TensorDims{FFOrdered{
                10_p,
                12_p,
            }},
            DataType::FLOAT,
        };

        tensor_guid_t input =
            b.create_input(input_shape, CreateGrad::YES, input_name);

        return b.computation_graph;
      }();

      layer_guid_t input_layer = get_layer_by_name(cg, input_name);

      std::map<TensorSlotName, tensor_guid_t> result =
          get_incoming_weights(cg, input_layer);
      std::map<TensorSlotName, tensor_guid_t> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("layer has inputs but no weights") {
      std::string layer_name = "my op";

      ComputationGraph cg = [&] {
        ComputationGraphBuilder b;

        TensorShape input_shape = TensorShape{
            TensorDims{FFOrdered{
                10_p,
                12_p,
            }},
            DataType::FLOAT,
        };

        tensor_guid_t input = b.create_input(input_shape, CreateGrad::YES);
        b.relu(input, layer_name);

        return b.computation_graph;
      }();

      layer_guid_t layer = get_layer_by_name(cg, layer_name);

      std::map<TensorSlotName, tensor_guid_t> result =
          get_incoming_weights(cg, layer);
      std::map<TensorSlotName, tensor_guid_t> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("layer has inputs and weights") {
      ComputationGraph cg = make_empty_computation_graph();

      TensorShape input_shape = TensorShape{
          TensorDims{FFOrdered{
              10_p,
              12_p,
          }},
          DataType::FLOAT,
      };

      auto make_layer_attrs = [](auto const &op_attrs) {
        return LayerAttrs{
            /*op_attrs=*/ComputationGraphOpAttrs{op_attrs},
            /*name=*/std::nullopt,
        };
      };

      LinearAttrs linear_attrs = LinearAttrs{
          /*out_channels=*/14_p,
          /*use_bias=*/true,
          /*data_type=*/DataType::FLOAT,
          /*activation=*/Activation::RELU,
          /*regularizer=*/std::nullopt,
      };

      InitializerAttrs zero_init = InitializerAttrs{ZeroInitializerAttrs{}};

      WeightAttrs projection_weight_attrs = WeightAttrs{
          /*tensor_shape=*/
          get_projection_shape(linear_attrs, input_shape),
          /*initializer=*/zero_init,
      };

      WeightAttrs bias_weight_attrs = WeightAttrs{
          /*tensor_shape=*/
          get_bias_shape(linear_attrs, input_shape),
          /*initializer=*/zero_init,
      };

      LayerAddedResult input_added = add_input_layer(cg, input_shape);
      tensor_guid_t t_input =
          require_only_key(input_added.outputs, TensorSlotName::OUTPUT);

      LayerAddedResult projection_weight_added =
          add_layer(cg, make_layer_attrs(projection_weight_attrs), {}, {});
      tensor_guid_t t_projection_weight = require_only_key(
          projection_weight_added.outputs, TensorSlotName::OUTPUT);

      LayerAddedResult bias_weight_added =
          add_layer(cg, make_layer_attrs(bias_weight_attrs), {}, {});
      tensor_guid_t t_bias_weight =
          require_only_key(bias_weight_added.outputs, TensorSlotName::OUTPUT);

      LayerAddedResult linear_added = add_layer(cg,
                                                make_layer_attrs(linear_attrs),
                                                {
                                                    {
                                                        TensorSlotName::INPUT,
                                                        t_input,
                                                    },
                                                },
                                                {
                                                    {
                                                        TensorSlotName::WEIGHT,
                                                        t_projection_weight,
                                                    },
                                                    {
                                                        TensorSlotName::BIAS,
                                                        t_bias_weight,
                                                    },
                                                });

      std::map<TensorSlotName, tensor_guid_t> result =
          get_incoming_weights(cg, linear_added.layer);
      std::map<TensorSlotName, tensor_guid_t> correct = {
          {
              TensorSlotName::WEIGHT,
              t_projection_weight,
          },
          {
              TensorSlotName::BIAS,
              t_bias_weight,
          },
      };

      CHECK(result == correct);
    }
  }
}
