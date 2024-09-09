#include "pcg/computation_graph.h"
#include "pcg/computation_graph_builder.h"
#include "utils/containers/get_only.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_incoming_inputs(ComputationGraph, layer_guid_t)") {
    SUBCASE("layer has no inputs") {
      std::string input_name = "input";
      ComputationGraph cg = [&] {
        ComputationGraphBuilder b;
        
        TensorShape input_shape = TensorShape{
          TensorDims{FFOrdered<size_t>{
            10, 12,
          }},
          DataType::FLOAT,
        };

        tensor_guid_t input = b.create_input(input_shape, CreateGrad::YES, input_name);

        return b.computation_graph;
      }();

      layer_guid_t input_layer = get_layer_by_name(cg, input_name);

      std::vector<tensor_guid_t> result = get_incoming_inputs(cg, input_layer);
      std::vector<tensor_guid_t> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("layer has inputs but no weights") {
      std::string layer_name = "my op";

      ComputationGraphBuilder b;
      
      TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered<size_t>{
          10, 12,
        }},
        DataType::FLOAT,
      };

      tensor_guid_t input = b.create_input(input_shape, CreateGrad::YES);
      b.relu(input, layer_name);

      ComputationGraph cg = b.computation_graph;

      layer_guid_t layer = get_layer_by_name(cg, layer_name);

      std::vector<tensor_guid_t> result = get_incoming_inputs(cg, layer);
      std::vector<tensor_guid_t> correct = {input};

      CHECK(result == correct);
    }

    SUBCASE("layer has inputs and weights") {
      std::string layer_name = "my op";

      ComputationGraphBuilder b;
      
      TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered<size_t>{
          10, 12,
        }},
        DataType::FLOAT,
      };

      tensor_guid_t input = b.create_input(input_shape, CreateGrad::YES);
      b.dense(input, 
              /*outDim=*/14,
              /*activation=*/Activation::RELU,
              /*use_bias=*/true,
              /*data_type=*/DataType::FLOAT,
              /*projection_initializer=*/std::nullopt,
              /*bias_initializer=*/std::nullopt,
              /*name=*/layer_name);

      ComputationGraph cg = b.computation_graph;

      layer_guid_t dense_layer = get_layer_by_name(cg, layer_name);

      std::vector<tensor_guid_t> result = get_incoming_inputs(cg, dense_layer);
      std::vector<tensor_guid_t> correct = {
        input,
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
          TensorDims{FFOrdered<size_t>{
            10, 12,
          }},
          DataType::FLOAT,
        };

        tensor_guid_t input = b.create_input(input_shape, CreateGrad::YES, input_name);

        return b.computation_graph;
      }();

      layer_guid_t input_layer = get_layer_by_name(cg, input_name);

      std::vector<tensor_guid_t> result = get_incoming_weights(cg, input_layer);
      std::vector<tensor_guid_t> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("layer has inputs but no weights") {
      std::string layer_name = "my op";

      ComputationGraph cg = [&] {
        ComputationGraphBuilder b;
        
        TensorShape input_shape = TensorShape{
          TensorDims{FFOrdered<size_t>{
            10, 12,
          }},
          DataType::FLOAT,
        };

        tensor_guid_t input = b.create_input(input_shape, CreateGrad::YES);
        b.relu(input, layer_name);

        return b.computation_graph;
      }();

      layer_guid_t layer = get_layer_by_name(cg, layer_name);

      std::vector<tensor_guid_t> result = get_incoming_weights(cg, layer);
      std::vector<tensor_guid_t> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("layer has inputs and weights") {
      std::string layer_name = "my op";
      std::string projection_name = "my projection weight";
      std::string bias_name = "my bias weight";

      ComputationGraph cg = [&] {
        ComputationGraphBuilder b;
        
        TensorShape input_shape = TensorShape{
          TensorDims{FFOrdered<size_t>{
            10, 12,
          }},
          DataType::FLOAT,
        };

        tensor_guid_t input = b.create_input(input_shape, CreateGrad::YES);
        b.dense(input, 
                /*outDim=*/14,
                /*activation=*/Activation::RELU,
                /*use_bias=*/true,
                /*data_type=*/DataType::FLOAT,
                /*projection_initializer=*/std::nullopt,
                /*bias_initializer=*/std::nullopt,
                /*name=*/layer_name,
                /*projection_name=*/projection_name,
                /*bias_name=*/bias_name);

        return b.computation_graph;
      }();

      layer_guid_t dense_layer = get_layer_by_name(cg, layer_name);

      layer_guid_t projection_weight_layer = get_layer_by_name(cg, projection_name);
      tensor_guid_t projection_weight = get_only(get_outgoing_tensors(cg, projection_weight_layer));

      layer_guid_t bias_weight_layer = get_layer_by_name(cg, bias_name);
      tensor_guid_t bias_weight = get_only(get_outgoing_tensors(cg, bias_weight_layer));

      std::vector<tensor_guid_t> result = get_incoming_weights(cg, dense_layer);
      std::vector<tensor_guid_t> correct = {
        projection_weight, 
        bias_weight,
      };

      CHECK(result == correct);
    }
  }
}
