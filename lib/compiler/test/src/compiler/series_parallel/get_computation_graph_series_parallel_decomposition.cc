#include "compiler/series_parallel/get_computation_graph_series_parallel_decomposition.h"
#include "models/inception_v3/inception_v3.h"
#include "models/split_test/split_test.h"
#include "models/transformer/transformer.h"
#include "pcg/computation_graph.h"
#include "pcg/computation_graph_builder.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE(
      "get_computation_graph_series_parallel_decomposition(ComputationGraph)") {
    SUBCASE("empty computation graph") {
      ComputationGraph cg = make_empty_computation_graph();

      std::optional<SeriesParallelDecomposition> result =
          get_computation_graph_series_parallel_decomposition(cg);
      // technically an empty graph is non-SP
      std::optional<SeriesParallelDecomposition> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("just a single input") {
      std::string input_layer_name = "my input";
      ComputationGraph cg = [&] {
        ComputationGraphBuilder b;

        TensorShape input_shape = TensorShape{TensorDims{FFOrdered<size_t>{
                                                  10,
                                                  12,
                                              }},
                                              DataType::FLOAT};
        b.create_input(input_shape, CreateGrad::YES, input_layer_name);

        return b.computation_graph;
      }();

      layer_guid_t input_layer = get_layer_by_name(cg, input_layer_name);

      std::optional<SeriesParallelDecomposition> result =
          get_computation_graph_series_parallel_decomposition(cg);
      std::optional<SeriesParallelDecomposition> correct =
          SeriesParallelDecomposition{input_layer.raw_node};

      CHECK(result == correct);
    }

    SUBCASE("single operator plus inputs and weights") {
      std::string input_layer_name = "my input";
      std::string projection_weights_layer_name = "my projection weights";
      std::string bias_weights_layer_name = "my bias weights";
      std::string operator_name = "my operator";
      ComputationGraph cg = [&] {
        ComputationGraphBuilder b;

        TensorShape input_shape = TensorShape{TensorDims{FFOrdered<size_t>{
                                                  10,
                                                  12,
                                              }},
                                              DataType::FLOAT};
        tensor_guid_t input =
            b.create_input(input_shape, CreateGrad::YES, input_layer_name);

        b.dense(input,
                /*outDim=*/14,
                /*activation=*/std::nullopt,
                /*use_bias=*/true,
                /*data_type=*/DataType::FLOAT,
                /*projection_initializer=*/std::nullopt,
                /*bias_initializer=*/std::nullopt,
                /*name=*/operator_name,
                /*projection_name=*/projection_weights_layer_name,
                /*bias_name=*/bias_weights_layer_name);

        return b.computation_graph;
      }();

      layer_guid_t input_layer = get_layer_by_name(cg, input_layer_name);
      layer_guid_t projection_weights_layer =
          get_layer_by_name(cg, projection_weights_layer_name);
      layer_guid_t bias_weights_layer =
          get_layer_by_name(cg, bias_weights_layer_name);
      layer_guid_t operator_layer = get_layer_by_name(cg, operator_name);

      std::optional<SeriesParallelDecomposition> result =
          get_computation_graph_series_parallel_decomposition(cg);
      std::optional<SeriesParallelDecomposition> correct =
          SeriesParallelDecomposition{SeriesSplit{
              ParallelSplit{
                  input_layer.raw_node,
                  projection_weights_layer.raw_node,
                  bias_weights_layer.raw_node,
              },
              operator_layer.raw_node,
          }};

      CHECK(result == correct);
    }

    SUBCASE("SP without weight nodes but non-SP with weight nodes") {
      // A minimal computation graph where without weights (w1 and w2) the
      // computation graph is series-parallel, but with weight nodes it is not
      //
      // w1   input   w2
      //  \   /   \   /
      //   op1     op2

      std::string w1_name = "w1";
      std::string input_name = "input";
      std::string w2_name = "w2";
      std::string op1_name = "op1";
      std::string op2_name = "op2";

      ComputationGraph cg = [&] {
        ComputationGraphBuilder b;

        TensorShape input_shape = TensorShape{
            TensorDims{FFOrdered<size_t>{
                10,
                12,
            }},
            DataType::FLOAT,
        };
        tensor_guid_t input =
            b.create_input(input_shape, CreateGrad::YES, input_name);

        b.dense(input,
                /*outDim=*/14,
                /*activation=*/std::nullopt,
                /*use_bias=*/false,
                /*data_type=*/DataType::FLOAT,
                /*projection_initializer=*/std::nullopt,
                /*bias_initializer=*/std::nullopt,
                /*name=*/op1_name,
                /*projection_name=*/w1_name);
        b.dense(input,
                /*outDim=*/14,
                /*activation=*/std::nullopt,
                /*use_bias=*/false,
                /*data_type=*/DataType::FLOAT,
                /*projection_initializer=*/std::nullopt,
                /*bias_initializer=*/std::nullopt,
                /*name=*/op2_name,
                /*projection_name=*/w2_name);

        return b.computation_graph;
      }();

      layer_guid_t w1 = get_layer_by_name(cg, w1_name);
      layer_guid_t input = get_layer_by_name(cg, input_name);
      layer_guid_t w2 = get_layer_by_name(cg, w2_name);
      layer_guid_t op1 = get_layer_by_name(cg, op1_name);
      layer_guid_t op2 = get_layer_by_name(cg, op2_name);

      std::optional<SeriesParallelDecomposition> result =
          get_computation_graph_series_parallel_decomposition(cg);
      std::optional<SeriesParallelDecomposition> correct =
          SeriesParallelDecomposition{SeriesSplit{
              ParallelSplit{
                  w1.raw_node,
                  input.raw_node,
                  w2.raw_node,
              },
              ParallelSplit{
                  op1.raw_node,
                  op2.raw_node,
              },
          }};
    }

    SUBCASE("SP with or without preprocessing, but preprocessing would SP "
            "decomposition") {
      // computation graph:
      //
      //  input1   input2
      //    |        |
      //   op1      op2

      std::string input1_name = "input1";
      std::string input2_name = "input2";
      std::string op1_name = "op1";
      std::string op2_name = "op2";

      ComputationGraph cg = [&] {
        ComputationGraphBuilder b;

        TensorShape input_shape = TensorShape{
            TensorDims{FFOrdered<size_t>{
                10,
                12,
            }},
            DataType::FLOAT,
        };
        tensor_guid_t input1 =
            b.create_input(input_shape, CreateGrad::YES, input1_name);
        tensor_guid_t input2 =
            b.create_input(input_shape, CreateGrad::YES, input2_name);

        b.relu(input1, op1_name);
        b.relu(input2, op2_name);

        return b.computation_graph;
      }();

      layer_guid_t input1 = get_layer_by_name(cg, input1_name);
      layer_guid_t input2 = get_layer_by_name(cg, input2_name);
      layer_guid_t op1 = get_layer_by_name(cg, op1_name);
      layer_guid_t op2 = get_layer_by_name(cg, op2_name);

      std::optional<SeriesParallelDecomposition> result =
          get_computation_graph_series_parallel_decomposition(cg);
      std::optional<SeriesParallelDecomposition> correct =
          SeriesParallelDecomposition{ParallelSplit{
              SeriesSplit{
                  input1.raw_node,
                  op1.raw_node,
              },
              SeriesSplit{
                  input2.raw_node,
                  op2.raw_node,
              },
          }};
    }

    SUBCASE("not SP with or without weight nodes") {
      // computation graph:
      //
      //    input1
      //     /  \
      //   op1  op2
      //    | \  |
      //    |  \ |
      //   op3  op4

      std::string input1_name = "input1";
      std::string op1_name = "op1";
      std::string op2_name = "op2";
      std::string op3_name = "op3";
      std::string op4_name = "op4";

      ComputationGraph cg = [&] {
        ComputationGraphBuilder b;

        TensorShape input_shape = TensorShape{
            TensorDims{FFOrdered<size_t>{
                10,
                12,
            }},
            DataType::FLOAT,
        };
        tensor_guid_t input1 =
            b.create_input(input_shape, CreateGrad::YES, input1_name);

        tensor_guid_t op1_output = b.relu(input1, op1_name);
        tensor_guid_t op2_output = b.relu(input1, op2_name);
        b.relu(op1_output, op3_name);
        b.add(op1_output, op2_output, op4_name);

        return b.computation_graph;
      }();

      layer_guid_t input1 = get_layer_by_name(cg, input1_name);
      layer_guid_t op1 = get_layer_by_name(cg, op1_name);
      layer_guid_t op2 = get_layer_by_name(cg, op2_name);
      layer_guid_t op3 = get_layer_by_name(cg, op3_name);
      layer_guid_t op4 = get_layer_by_name(cg, op4_name);

      std::optional<SeriesParallelDecomposition> result =
          get_computation_graph_series_parallel_decomposition(cg);
      std::optional<SeriesParallelDecomposition> correct = std::nullopt;
    }

    SUBCASE("real models") {
      SUBCASE("split_test") {
        ComputationGraph cg =
            get_split_test_computation_graph(/*batch_size=*/8);

        std::optional<SeriesParallelDecomposition> sp_decomposition =
            get_computation_graph_series_parallel_decomposition(cg);

        CHECK(sp_decomposition.has_value());
      }

      SUBCASE("transformer") {
        ComputationGraph cg =
            get_transformer_computation_graph(get_default_transformer_config());

        std::optional<SeriesParallelDecomposition> sp_decomposition =
            get_computation_graph_series_parallel_decomposition(cg);

        CHECK(sp_decomposition.has_value());
      }

      SUBCASE("inception_v3") {
        ComputationGraph cg = get_inception_v3_computation_graph(
            get_default_inception_v3_training_config());

        std::optional<SeriesParallelDecomposition> sp_decomposition =
            get_computation_graph_series_parallel_decomposition(cg);

        CHECK(sp_decomposition.has_value());
      }
    }
  }

  TEST_CASE("render_preprocessed_computation_graph_for_sp_decomposition("
            "ComputationGraph)") {
    // currently there's not really a good way to test this, and its arguable
    // how much its output really should be validated as its primarily for
    // visualization and so there's not really a strict definition of
    // correctness, so for now we just run it on some models and make sure it
    // doesn't crash. Don't use this as an example.

    SUBCASE("basic single-operator model") {
      ComputationGraph cg = [&] {
        ComputationGraphBuilder b;

        TensorShape input_shape = TensorShape{TensorDims{FFOrdered<size_t>{
                                                  10,
                                                  12,
                                              }},
                                              DataType::FLOAT};
        tensor_guid_t input = b.create_input(input_shape, CreateGrad::YES);

        b.dense(input, /*outDim=*/14);

        return b.computation_graph;
      }();

      std::string result =
          render_preprocessed_computation_graph_for_sp_decomposition(cg);
    }

    SUBCASE("split_test") {
      ComputationGraph cg = get_split_test_computation_graph(/*batch_size=*/8);

      std::string result =
          render_preprocessed_computation_graph_for_sp_decomposition(cg);
    }

    SUBCASE("transformer") {
      ComputationGraph cg =
          get_transformer_computation_graph(get_default_transformer_config());

      std::string result =
          render_preprocessed_computation_graph_for_sp_decomposition(cg);
    }
  }
}
