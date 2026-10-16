#include "compiler/series_parallel/computation_graph/get_computation_graph_series_parallel_decomposition.h"
#include "models/bert/bert.h"
#include "models/candle_uno/candle_uno.h"
#include "models/dlrm/dlrm.h"
#include "models/inception_v3/inception_v3.h"
#include "models/split_test/split_test.h"
#include "models/transformer/transformer.h"
#include "models/yolov10/yolov10.h"
#include "op-attrs/ops/linear.h"
#include "pcg/computation_graph.h"
#include "pcg/computation_graph_builder.h"
#include "utils/containers/get_only.h"
#include "utils/containers/require_only_key.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE(
      "get_computation_graph_series_parallel_decomposition(ComputationGraph)") {
    auto make_layer_attrs = [](auto const &op_attrs) {
      return LayerAttrs{
          /*op_attrs=*/ComputationGraphOpAttrs{op_attrs},
          /*name=*/std::nullopt,
      };
    };

    ComputationGraph cg = make_empty_computation_graph();

    SUBCASE("empty computation graph") {
      std::optional<SeriesParallelDecomposition> result =
          get_computation_graph_series_parallel_decomposition(cg);
      // technically an empty graph is non-SP
      std::optional<SeriesParallelDecomposition> correct = std::nullopt;

      CHECK(result == correct);
    }

    InitializerAttrs zero_init = InitializerAttrs{ZeroInitializerAttrs{}};

    TensorShape input_shape = TensorShape{TensorDims{
                                              FFOrdered{
                                                  10_p,
                                                  12_p,
                                              },
                                          },
                                          DataType::FLOAT};

    InputAttrs input_attrs = InputAttrs{
        /*shape=*/input_shape,
    };

    SUBCASE("just a single input") {
      LayerAddedResult input_added =
          add_layer(cg, make_layer_attrs(input_attrs), {}, {});

      std::optional<SeriesParallelDecomposition> result =
          get_computation_graph_series_parallel_decomposition(cg);
      std::optional<SeriesParallelDecomposition> correct =
          SeriesParallelDecomposition{input_added.layer.raw_node};

      CHECK(result == correct);
    }

    SUBCASE("single operator plus inputs and weights") {
      LinearAttrs linear_attrs = LinearAttrs{
          /*out_channels=*/14_p,
          /*use_bias=*/true,
          /*data_type=*/DataType::FLOAT,
          /*activation=*/std::nullopt,
          /*regularizer=*/std::nullopt,
      };

      TensorShape projection_weight_shape =
          linear_get_projection_shape(linear_attrs, input_shape);

      TensorShape bias_weight_shape =
          linear_get_bias_shape(linear_attrs, input_shape);

      WeightAttrs projection_weight_attrs = WeightAttrs{
          /*tensor_shape=*/projection_weight_shape,
          /*initializer=*/zero_init,
      };

      WeightAttrs bias_weight_attrs = WeightAttrs{
          /*tensor_shape=*/bias_weight_shape,
          /*initializer=*/zero_init,
      };

      LayerAddedResult input_added =
          add_layer(cg, make_layer_attrs(input_attrs), {}, {});
      tensor_guid_t t_input =
          require_only_key(input_added.outputs, TensorSlotName::OUTPUT);

      LayerAddedResult projection_weight_added =
          add_layer(cg, make_layer_attrs(projection_weight_attrs), {}, {});
      tensor_guid_t t_projection = require_only_key(
          projection_weight_added.outputs, TensorSlotName::OUTPUT);

      LayerAddedResult bias_weight_added =
          add_layer(cg, make_layer_attrs(bias_weight_attrs), {}, {});
      tensor_guid_t t_bias =
          require_only_key(bias_weight_added.outputs, TensorSlotName::OUTPUT);

      LayerAddedResult linear_added = add_layer(
          /*computation_graph=*/cg,
          /*attrs=*/make_layer_attrs(linear_attrs),
          /*inputs=*/
          {
              {
                  TensorSlotName::INPUT,
                  t_input,
              },
          },
          /*weights=*/
          {
              {
                  TensorSlotName::WEIGHT,
                  t_projection,
              },
              {
                  TensorSlotName::BIAS,
                  t_bias,
              },
          });

      std::optional<SeriesParallelDecomposition> result =
          get_computation_graph_series_parallel_decomposition(cg);
      std::optional<SeriesParallelDecomposition> correct =
          SeriesParallelDecomposition{SeriesSplit{{
              ParallelSplit{{
                  input_added.layer.raw_node,
                  projection_weight_added.layer.raw_node,
                  bias_weight_added.layer.raw_node,
              }},
              linear_added.layer.raw_node,
          }}};

      CHECK(result == correct);
    }

    SUBCASE("SP without weight nodes but non-SP with weight nodes") {
      // A minimal computation graph where without weights (w1 and w2) the
      // computation graph is series-parallel, but with weight nodes it is not
      //
      // w1   input   w2
      //  \   /   \   /
      //   op1     op2

      LinearAttrs linear_attrs = LinearAttrs{
          /*out_channels=*/14_p,
          /*use_bias=*/false,
          /*data_type=*/DataType::FLOAT,
          /*activation=*/std::nullopt,
          /*regularizer=*/std::nullopt,
      };

      TensorShape projection_weight_shape =
          linear_get_projection_shape(linear_attrs, input_shape);

      WeightAttrs projection_weight_attrs = WeightAttrs{
          /*tensor_shape=*/projection_weight_shape,
          /*initializer=*/zero_init,
      };

      LayerAddedResult input_added =
          add_layer(cg, make_layer_attrs(input_attrs), {}, {});
      tensor_guid_t t_input =
          require_only_key(input_added.outputs, TensorSlotName::OUTPUT);

      LayerAddedResult w1_added =
          add_layer(cg, make_layer_attrs(projection_weight_attrs), {}, {});
      tensor_guid_t t_w1 =
          require_only_key(w1_added.outputs, TensorSlotName::OUTPUT);

      LayerAddedResult w2_added =
          add_layer(cg, make_layer_attrs(projection_weight_attrs), {}, {});
      tensor_guid_t t_w2 =
          require_only_key(w2_added.outputs, TensorSlotName::OUTPUT);

      LayerAddedResult op1_added = add_layer(
          /*computation_graph=*/cg,
          /*layer_attrs=*/make_layer_attrs(linear_attrs),
          /*input=*/
          {
              {
                  TensorSlotName::INPUT,
                  t_input,
              },
          },
          /*weights=*/
          {
              {
                  TensorSlotName::WEIGHT,
                  t_w1,
              },
          });

      LayerAddedResult op2_added = add_layer(
          /*computation_graph=*/cg,
          /*layer_attrs=*/make_layer_attrs(linear_attrs),
          /*inputs=*/
          {
              {
                  TensorSlotName::INPUT,
                  t_input,
              },
          },
          /*weights=*/
          {
              {
                  TensorSlotName::WEIGHT,
                  t_w2,
              },
          });

      std::optional<SeriesParallelDecomposition> result =
          get_computation_graph_series_parallel_decomposition(cg);
      std::optional<SeriesParallelDecomposition> correct =
          SeriesParallelDecomposition{SeriesSplit{{
              ParallelSplit{{
                  w1_added.layer.raw_node,
                  input_added.layer.raw_node,
                  w2_added.layer.raw_node,
              }},
              ParallelSplit{{
                  op1_added.layer.raw_node,
                  op2_added.layer.raw_node,
              }},
          }}};

      CHECK(result == correct);
    }

    ElementUnaryAttrs relu_attrs = ElementUnaryAttrs{
        /*op_type=*/ElementUnaryOp::RELU,
        /*scalar=*/std::nullopt,
    };

    SUBCASE("SP with or without preprocessing, but preprocessing would change "
            "resulting SP "
            "decomposition") {
      // computation graph:
      //
      //  input1   input2
      //    |        |
      //   op1      op2

      LayerAddedResult input1_added =
          add_layer(cg, make_layer_attrs(input_attrs), {}, {});
      tensor_guid_t t_input1 =
          require_only_key(input1_added.outputs, TensorSlotName::OUTPUT);

      LayerAddedResult input2_added =
          add_layer(cg, make_layer_attrs(input_attrs), {}, {});
      tensor_guid_t t_input2 =
          require_only_key(input2_added.outputs, TensorSlotName::OUTPUT);

      LayerAddedResult op1_added = add_layer(
          /*computation_graph=*/cg,
          /*layer_attrs=*/make_layer_attrs(relu_attrs),
          /*inputs=*/
          {
              {
                  TensorSlotName::INPUT,
                  t_input1,
              },
          },
          /*weights=*/{});

      LayerAddedResult op2_added = add_layer(
          /*computation_graph=*/cg,
          /*layer_attrs=*/make_layer_attrs(relu_attrs),
          /*inputs=*/
          {
              {
                  TensorSlotName::INPUT,
                  t_input2,
              },
          },
          /*weights=*/{});

      std::optional<SeriesParallelDecomposition> result =
          get_computation_graph_series_parallel_decomposition(cg);
      std::optional<SeriesParallelDecomposition> correct =
          SeriesParallelDecomposition{ParallelSplit{{
              SeriesSplit{{
                  input1_added.layer.raw_node,
                  op1_added.layer.raw_node,
              }},
              SeriesSplit{{
                  input2_added.layer.raw_node,
                  op2_added.layer.raw_node,
              }},
          }}};

      CHECK(result == correct);
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

      LayerAddedResult input1_added =
          add_layer(cg, make_layer_attrs(input_attrs), {}, {});
      tensor_guid_t t_input1 =
          require_only_key(input1_added.outputs, TensorSlotName::OUTPUT);

      ElementBinaryAttrs ew_add_attrs = ElementBinaryAttrs{
          /*op=*/ElementBinaryOp::ADD,
          /*compute_type=*/DataType::FLOAT,
          /*should_broadcast_lhs=*/false,
          /*should_broadcast_rhs=*/false,
      };

      LayerAddedResult op1_added = add_layer(
          /*computation_graph=*/cg,
          /*layer_attrs=*/make_layer_attrs(relu_attrs),
          /*inputs=*/
          {
              {
                  TensorSlotName::INPUT,
                  t_input1,
              },
          },
          /*weights=*/{});
      tensor_guid_t t_op1 =
          require_only_key(op1_added.outputs, TensorSlotName::OUTPUT);

      LayerAddedResult op2_added = add_layer(
          /*computation_graph=*/cg,
          /*layer_attrs=*/make_layer_attrs(relu_attrs),
          /*inputs=*/
          {
              {
                  TensorSlotName::INPUT,
                  t_input1,
              },
          },
          /*weights=*/{});
      tensor_guid_t t_op2 =
          require_only_key(op2_added.outputs, TensorSlotName::OUTPUT);

      LayerAddedResult op3_added = add_layer(
          /*computation_graph=*/cg,
          /*layer_attrs=*/make_layer_attrs(relu_attrs),
          /*inputs=*/
          {
              {
                  TensorSlotName::INPUT,
                  t_op1,
              },
          },
          /*weights=*/{});

      LayerAddedResult op4_added = add_layer(
          /*computation_graph=*/cg,
          /*layer_attrs=*/make_layer_attrs(ew_add_attrs),
          /*inputs=*/
          {
              {
                  TensorSlotName::LHS_INPUT,
                  t_op1,
              },
              {
                  TensorSlotName::RHS_INPUT,
                  t_op2,
              },
          },
          /*weights=*/{});

      std::optional<SeriesParallelDecomposition> result =
          get_computation_graph_series_parallel_decomposition(cg);
      std::optional<SeriesParallelDecomposition> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("real models") {
      SUBCASE("split_test") {
        ComputationGraph cg =
            get_split_test_computation_graph(/*batch_size=*/8_p);

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

      SUBCASE("candle_uno") {
        ComputationGraph cg =
            get_candle_uno_computation_graph(get_default_candle_uno_config());

        std::optional<SeriesParallelDecomposition> sp_decomposition =
            get_computation_graph_series_parallel_decomposition(cg);

        CHECK(sp_decomposition.has_value());
      }

      SUBCASE("bert") {
        ComputationGraph cg =
            get_bert_computation_graph(get_default_bert_config());

        std::optional<SeriesParallelDecomposition> sp_decomposition =
            get_computation_graph_series_parallel_decomposition(cg);

        CHECK(sp_decomposition.has_value());
      }

      SUBCASE("dlrm") {
        ComputationGraph cg =
            get_dlrm_computation_graph(get_default_dlrm_config());

        std::optional<SeriesParallelDecomposition> sp_decomposition =
            get_computation_graph_series_parallel_decomposition(cg);

        CHECK(sp_decomposition.has_value());
      }

      SUBCASE("yolov10x") {
        ComputationGraph cg = get_yolov10_computation_graph(get_yolov10x_config(
            /*batch_size=*/8_p,
            /*end2end=*/false));

        std::optional<SeriesParallelDecomposition> sp_decomposition =
            get_computation_graph_series_parallel_decomposition(cg);

        CHECK(!sp_decomposition.has_value());
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

        TensorShape input_shape = TensorShape{TensorDims{FFOrdered{
                                                  10_p,
                                                  12_p,
                                              }},
                                              DataType::FLOAT};
        tensor_guid_t input = b.create_input(input_shape, CreateGrad::YES);

        b.dense(input, /*outDim=*/14_p);

        return b.computation_graph;
      }();

      std::string result =
          render_preprocessed_computation_graph_for_sp_decomposition(cg);
    }

    SUBCASE("split_test") {
      ComputationGraph cg =
          get_split_test_computation_graph(/*batch_size=*/8_p);

      std::string result =
          render_preprocessed_computation_graph_for_sp_decomposition(cg);
    }

    SUBCASE("transformer") {
      ComputationGraph cg =
          get_transformer_computation_graph(get_default_transformer_config());

      std::string result =
          render_preprocessed_computation_graph_for_sp_decomposition(cg);
    }

    SUBCASE("inception_v3") {
      ComputationGraph cg = get_inception_v3_computation_graph(
          get_default_inception_v3_training_config());

      std::string result =
          render_preprocessed_computation_graph_for_sp_decomposition(cg);
    }

    SUBCASE("candle_uno") {
      ComputationGraph cg =
          get_candle_uno_computation_graph(get_default_candle_uno_config());

      std::string result =
          render_preprocessed_computation_graph_for_sp_decomposition(cg);
    }

    SUBCASE("bert") {
      ComputationGraph cg =
          get_bert_computation_graph(get_default_bert_config());

      std::string result =
          render_preprocessed_computation_graph_for_sp_decomposition(cg);
    }

    SUBCASE("dlrm") {
      ComputationGraph cg =
          get_dlrm_computation_graph(get_default_dlrm_config());

      std::string result =
          render_preprocessed_computation_graph_for_sp_decomposition(cg);
    }

    SUBCASE("yolov10x") {
      ComputationGraph cg = get_yolov10_computation_graph(get_yolov10x_config(
          /*batch_size=*/8_p,
          /*end2end=*/false));

      std::string result =
          render_preprocessed_computation_graph_for_sp_decomposition(cg);
    }
  }
}
