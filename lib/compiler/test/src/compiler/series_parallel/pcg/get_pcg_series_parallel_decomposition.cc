#include "compiler/series_parallel/pcg/get_pcg_series_parallel_decomposition.h"
#include "op-attrs/element_binary_op.dtg.h"
#include "op-attrs/ops/embedding.h"
#include "op-attrs/ops/linear.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "utils/containers/get_only.h"
#include "utils/containers/require_only_key.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_pcg_series_parallel_decomposition(ParallelComputationGraph "
            "const &)") {
    SUBCASE("empty pcg") {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      std::optional<SeriesParallelDecomposition> result =
          get_pcg_series_parallel_decomposition(pcg);
      // technically an empty graph is non-SP
      std::optional<SeriesParallelDecomposition> correct = std::nullopt;

      CHECK(result == correct);
    }

    TensorShape input_shape = TensorShape{TensorDims{FFOrdered{
                                              10_p,
                                              12_p,
                                          }},
                                          DataType::FLOAT};
    InitializerAttrs zero_init = InitializerAttrs{ZeroInitializerAttrs{}};

    auto make_layer_attrs = [](auto const &op_attrs) -> ParallelLayerAttrs {
      return ParallelLayerAttrs{
          /*op_attrs=*/PCGOperatorAttrs{op_attrs},
          /*name=*/std::nullopt,
      };
    };

    SUBCASE("just a single input") {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      ParallelLayerAddedResult added = pcg_add_input_layer(pcg, input_shape);
      parallel_layer_guid_t input_layer = added.parallel_layer;

      std::optional<SeriesParallelDecomposition> result =
          get_pcg_series_parallel_decomposition(pcg);
      std::optional<SeriesParallelDecomposition> correct =
          SeriesParallelDecomposition{input_layer.raw_graph_node};

      CHECK(result == correct);
    }

    SUBCASE("single operator plus inputs and weights") {
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      ParallelLayerAddedResult input_added =
          pcg_add_input_layer(pcg, input_shape);
      parallel_tensor_guid_t t_input =
          require_only_key(input_added.outputs, TensorSlotName::OUTPUT);

      LinearAttrs linear_attrs = LinearAttrs{
          /*out_channels=*/14_p,
          /*use_bias=*/true,
          /*data_type=*/DataType::FLOAT,
          /*activation=*/Activation::RELU,
          /*regularizer=*/std::nullopt,
      };

      TensorShape projection_weights_shape =
          throw_if_unexpected(get_projection_shape(linear_attrs, input_shape));
      TensorShape bias_weights_shape =
          throw_if_unexpected(get_bias_shape(linear_attrs, input_shape));

      WeightAttrs projection_weight_attrs = WeightAttrs{
          /*shape=*/projection_weights_shape,
          /*initializer=*/InitializerAttrs{ZeroInitializerAttrs{}},
      };
      ParallelLayerAddedResult projection_weights_added = add_parallel_layer(
          pcg,
          /*layer_attrs=*/make_layer_attrs(projection_weight_attrs),
          /*inputs=*/{},
          /*weights=*/{});
      parallel_tensor_guid_t t_projection_weights = require_only_key(
          projection_weights_added.outputs, TensorSlotName::OUTPUT);

      WeightAttrs bias_weight_attrs = WeightAttrs{
          /*shape=*/bias_weights_shape,
          /*initializer=*/InitializerAttrs{ZeroInitializerAttrs{}},
      };
      ParallelLayerAddedResult bias_weights_added = add_parallel_layer(
          pcg,
          /*layer_attrs=*/make_layer_attrs(bias_weight_attrs),
          /*inputs=*/{},
          /*weights=*/{});
      parallel_tensor_guid_t t_bias_weights =
          require_only_key(bias_weights_added.outputs, TensorSlotName::OUTPUT);

      ParallelLayerAddedResult linear_added =
          add_parallel_layer(pcg,
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
                                     t_projection_weights,
                                 },
                                 {
                                     TensorSlotName::BIAS,
                                     t_bias_weights,
                                 },
                             });

      std::optional<SeriesParallelDecomposition> result =
          get_pcg_series_parallel_decomposition(pcg);
      std::optional<SeriesParallelDecomposition> correct =
          SeriesParallelDecomposition{SeriesSplit{{
              ParallelSplit{{
                  input_added.parallel_layer.raw_graph_node,
                  projection_weights_added.parallel_layer.raw_graph_node,
                  bias_weights_added.parallel_layer.raw_graph_node,
              }},
              linear_added.parallel_layer.raw_graph_node,
          }}};

      CHECK(result == correct);
    }

    SUBCASE("SP without weight nodes but non-SP with weight nodes (parallel op "
            "chain following is not necessary)") {
      /**
       * A minimal computation graph where without weights (w1 and w2) the
       * computation graph is series-parallel, but with weight nodes it is not,
       * but parallel op chain following is not necessary
       * (in this case because there are no parallel ops involved)
       *
       * w1   input   w2
       *  \   /   \   /
       *   op1     op2
       */

      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      InputAttrs input_attrs = InputAttrs{
          /*tensor_shape=*/input_shape,
      };

      LinearAttrs linear_attrs = LinearAttrs{
          /*out_channels=*/14_p,
          /*use_bias=*/false,
          /*data_type=*/DataType::FLOAT,
          /*activation=*/std::nullopt,
          /*regularizer=*/std::nullopt,
      };

      TensorShape projection_weight_shape =
          throw_if_unexpected(get_projection_shape(linear_attrs, input_shape));

      WeightAttrs projection_weight_attrs = WeightAttrs{
          /*tensor_shape=*/projection_weight_shape,
          /*initializer=*/zero_init,
      };

      ParallelLayerAddedResult input_added =
          add_parallel_layer(pcg, make_layer_attrs(input_attrs), {}, {});
      parallel_tensor_guid_t t_input =
          require_only_key(input_added.outputs, TensorSlotName::OUTPUT);

      ParallelLayerAddedResult w1_added = add_parallel_layer(
          pcg, make_layer_attrs(projection_weight_attrs), {}, {});
      parallel_tensor_guid_t t_w1 =
          require_only_key(w1_added.outputs, TensorSlotName::OUTPUT);

      ParallelLayerAddedResult w2_added = add_parallel_layer(
          pcg, make_layer_attrs(projection_weight_attrs), {}, {});
      parallel_tensor_guid_t t_w2 =
          require_only_key(w2_added.outputs, TensorSlotName::OUTPUT);

      ParallelLayerAddedResult op1_added =
          add_parallel_layer(pcg,
                             make_layer_attrs(linear_attrs),
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
                                     t_w1,
                                 },
                             });

      ParallelLayerAddedResult op2_added =
          add_parallel_layer(pcg,
                             make_layer_attrs(linear_attrs),
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
          get_pcg_series_parallel_decomposition(pcg);
      std::optional<SeriesParallelDecomposition> correct =
          SeriesParallelDecomposition{SeriesSplit{{
              ParallelSplit{{
                  w1_added.parallel_layer.raw_graph_node,
                  input_added.parallel_layer.raw_graph_node,
                  w2_added.parallel_layer.raw_graph_node,
              }},
              ParallelSplit{{
                  op1_added.parallel_layer.raw_graph_node,
                  op2_added.parallel_layer.raw_graph_node,
              }},
          }}};

      CHECK(result == correct);
    }

    SUBCASE("SP without weight nodes but non-SP with weight node (parallel op "
            "chain following necessary)") {

      /**
       * A minimal computation graph where without weights (w1 and w2) the
       * computation graph is series-parallel, but with weight nodes it is not
       * and parallel op chain following is necessary
       *
       * w1   input   w2
       *  |    |       |
       *  |    p2     p4
       *  |    |       |
       * p1    p3     p5
       *  |    |       |
       *  |    |\     /
       *  |  op0 \    |
       *  \   /   |  /
       *   op1    op2
       */

      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      TensorShape input_shape = TensorShape{
          TensorDims{FFOrdered{
              12_p,
              10_p,
          }},
          DataType::FLOAT,
      };

      ParallelLayerAddedResult input_added =
          pcg_add_input_layer(pcg, input_shape);
      parallel_layer_guid_t layer_input = input_added.parallel_layer;
      parallel_tensor_guid_t t_input =
          require_only_key(input_added.outputs, TensorSlotName::OUTPUT);

      RepartitionAttrs p2_attrs = RepartitionAttrs{
          /*repartition_dim=*/ff_dim_t{0_n},
          /*repartition_degree=*/3_ge2,
      };
      ParallelLayerAddedResult p2_added =
          add_parallel_layer(pcg,
                             make_layer_attrs(p2_attrs),
                             {{TensorSlotName::INPUT, t_input}},
                             {});
      parallel_tensor_guid_t t_p2 =
          require_only_key(p2_added.outputs, TensorSlotName::OUTPUT);

      ParallelLayerAttrs p3_attrs = ParallelLayerAttrs{
          PCGOperatorAttrs{RepartitionAttrs{
              /*repartition_dim=*/ff_dim_t{1_n},
              /*repartition_degree=*/2_ge2,
          }},
          /*name=*/std::nullopt,
      };
      ParallelLayerAddedResult p3_added = add_parallel_layer(
          pcg, p3_attrs, {{TensorSlotName::INPUT, t_p2}}, {});
      parallel_tensor_guid_t t_p3 =
          require_only_key(p3_added.outputs, TensorSlotName::OUTPUT);

      CastAttrs op0_attrs = CastAttrs{
          /*dtype=*/DataType::INT32,
      };
      ParallelLayerAddedResult op0_added =
          add_parallel_layer(pcg,
                             make_layer_attrs(op0_attrs),
                             {{TensorSlotName::INPUT, t_p3}},
                             {});
      parallel_tensor_guid_t t_op0 =
          require_only_key(op0_added.outputs, TensorSlotName::OUTPUT);

      EmbeddingAttrs op1_attrs = EmbeddingAttrs{
          /*num_entires=*/100_p,
          /*out_channels=*/22_p,
          /*aggr=*/AggregateOp::SUM,
          /*data_type=*/DataType::FLOAT,
      };

      TensorShape casted_input_shape =
          get_reduced_shape(get_parallel_tensor_shape(pcg, t_op0));

      WeightAttrs w1_attrs = WeightAttrs{
          /*tensor_shape=*/throw_if_unexpected(
              get_weights_shape(op1_attrs, casted_input_shape)),
          /*initializer=*/InitializerAttrs{ZeroInitializerAttrs{}},
      };
      ParallelLayerAddedResult w1_added =
          add_parallel_layer(pcg, make_layer_attrs(w1_attrs), {}, {});
      parallel_tensor_guid_t t_w1 =
          require_only_key(w1_added.outputs, TensorSlotName::OUTPUT);

      ReplicateAttrs p1_attrs = ReplicateAttrs{
          /*replicate_degree=*/6_ge2,
      };
      ParallelLayerAddedResult p1_added = add_parallel_layer(
          pcg, make_layer_attrs(p1_attrs), {{TensorSlotName::INPUT, t_w1}}, {});
      parallel_tensor_guid_t t_p1 =
          require_only_key(p1_added.outputs, TensorSlotName::OUTPUT);

      ParallelLayerAddedResult op1_added = add_parallel_layer(
          /*pcg=*/pcg,
          /*layer_attrs=*/make_layer_attrs(op1_attrs),
          /*inputs=*/
          {
              {
                  TensorSlotName::INPUT,
                  t_op0,
              },
          },
          /*weights=*/
          {
              {
                  TensorSlotName::WEIGHT,
                  t_p1,
              },
          });

      LinearAttrs op2_attrs = LinearAttrs{
          /*out_channels=*/14_p,
          /*use_bias=*/false,
          /*data_type=*/DataType::FLOAT,
          /*activation=*/std::nullopt,
          /*regularizer=*/std::nullopt,
      };

      WeightAttrs w2_attrs = WeightAttrs{
          /*tensor_shape=*/throw_if_unexpected(
              get_projection_shape(op2_attrs, input_shape)),
          /*initializer=*/InitializerAttrs{ZeroInitializerAttrs{}},
      };
      ParallelLayerAddedResult w2_added =
          add_parallel_layer(pcg, make_layer_attrs(w2_attrs), {}, {});
      parallel_tensor_guid_t t_w2 =
          require_only_key(w2_added.outputs, TensorSlotName::OUTPUT);

      ReplicateAttrs p4_attrs = ReplicateAttrs{
          /*replicate_degree=*/3_ge2,
      };
      ParallelLayerAddedResult p4_added = add_parallel_layer(
          pcg, make_layer_attrs(p4_attrs), {{TensorSlotName::INPUT, t_w2}}, {});
      parallel_tensor_guid_t t_p4 =
          require_only_key(p4_added.outputs, TensorSlotName::OUTPUT);

      RepartitionAttrs p5_attrs = RepartitionAttrs{
          /*repartition_dim=*/ff_dim_t{1_n},
          /*repartition_degree=*/2_ge2,
      };
      ParallelLayerAddedResult p5_added = add_parallel_layer(
          pcg, make_layer_attrs(p5_attrs), {{TensorSlotName::INPUT, t_p4}}, {});
      parallel_tensor_guid_t t_p5 =
          require_only_key(p5_added.outputs, TensorSlotName::OUTPUT);

      ParallelLayerAddedResult op2_added = add_parallel_layer(
          /*pcg=*/pcg,
          /*layer_attrs=*/make_layer_attrs(op2_attrs),
          /*inputs=*/
          {
              {
                  TensorSlotName::INPUT,
                  t_p3,
              },
          },
          /*weights=*/
          {
              {
                  TensorSlotName::WEIGHT,
                  t_p5,
              },
          });

      std::optional<SeriesParallelDecomposition> result =
          get_pcg_series_parallel_decomposition(pcg);
      std::optional<SeriesParallelDecomposition> correct =
          SeriesParallelDecomposition{SeriesSplit{{
              ParallelSplit{{
                  SeriesSplit{{
                      w1_added.parallel_layer.raw_graph_node,
                      p1_added.parallel_layer.raw_graph_node,
                  }},
                  SeriesSplit{{
                      input_added.parallel_layer.raw_graph_node,
                      p2_added.parallel_layer.raw_graph_node,
                      p3_added.parallel_layer.raw_graph_node,
                  }},
                  SeriesSplit{{
                      w2_added.parallel_layer.raw_graph_node,
                      p4_added.parallel_layer.raw_graph_node,
                      p5_added.parallel_layer.raw_graph_node,
                  }},
              }},
              ParallelSplit{{
                  SeriesSplit{{
                      op0_added.parallel_layer.raw_graph_node,
                      op1_added.parallel_layer.raw_graph_node,
                  }},
                  op2_added.parallel_layer.raw_graph_node,
              }},
          }}};

      CHECK(result == correct);
    }

    ParallelComputationGraph pcg = empty_parallel_computation_graph();
    InputAttrs input_attrs = InputAttrs{
        /*tensor_shape=*/input_shape,
    };
    ElementUnaryAttrs relu_attrs = ElementUnaryAttrs{
        /*op_type=*/ElementUnaryOp::RELU,
        /*scalar=*/std::nullopt,
    };

    SUBCASE("SP with or without preprocessing, but preprocessing would change "
            "resulting SP "
            "decomposition") {

      /**
       * parallel computation graph:
       *
       *  input1   input2
       *    |        |
       *   op1      op2
       */

      ParallelLayerAddedResult input1_added =
          add_parallel_layer(pcg, make_layer_attrs(input_attrs), {}, {});
      parallel_tensor_guid_t t_input1 =
          require_only_key(input1_added.outputs, TensorSlotName::OUTPUT);

      ParallelLayerAddedResult input2_added =
          add_parallel_layer(pcg, make_layer_attrs(input_attrs), {}, {});
      parallel_tensor_guid_t t_input2 =
          require_only_key(input2_added.outputs, TensorSlotName::OUTPUT);

      ParallelLayerAddedResult op1_added =
          add_parallel_layer(pcg,
                             make_layer_attrs(relu_attrs),
                             {{TensorSlotName::INPUT, t_input1}},
                             {});

      ParallelLayerAddedResult op2_added =
          add_parallel_layer(pcg,
                             make_layer_attrs(relu_attrs),
                             {{TensorSlotName::INPUT, t_input2}},
                             {});

      std::optional<SeriesParallelDecomposition> result =
          get_pcg_series_parallel_decomposition(pcg);
      std::optional<SeriesParallelDecomposition> correct =
          SeriesParallelDecomposition{ParallelSplit{{
              SeriesSplit{{
                  input1_added.parallel_layer.raw_graph_node,
                  op1_added.parallel_layer.raw_graph_node,
              }},
              SeriesSplit{{
                  input2_added.parallel_layer.raw_graph_node,
                  op2_added.parallel_layer.raw_graph_node,
              }},
          }}};

      CHECK(result == correct);
    }

    SUBCASE("not SP with or without weight nodes") {

      /**
       * parallel computation graph:
       *
       *    input1
       *     /  \
       *   op1  op2
       *    | \  |
       *    |  \ |
       *   op3  op4
       */

      ParallelLayerAddedResult input1_added =
          add_parallel_layer(pcg, make_layer_attrs(input_attrs), {}, {});
      parallel_tensor_guid_t t_input1 =
          require_only_key(input1_added.outputs, TensorSlotName::OUTPUT);

      ElementBinaryAttrs ew_add_attrs = ElementBinaryAttrs{
          /*op=*/ElementBinaryOp::ADD,
          /*compute_type=*/DataType::FLOAT,
          /*should_broadcast_lhs=*/false,
          /*should_broadcast_rhs=*/false,
      };

      ParallelLayerAddedResult op1_added =
          add_parallel_layer(pcg,
                             make_layer_attrs(relu_attrs),
                             {{TensorSlotName::INPUT, t_input1}},
                             {});
      parallel_tensor_guid_t t_op1 =
          require_only_key(op1_added.outputs, TensorSlotName::OUTPUT);

      ParallelLayerAddedResult op2_added =
          add_parallel_layer(pcg,
                             make_layer_attrs(relu_attrs),
                             {{TensorSlotName::INPUT, t_input1}},
                             {});
      parallel_tensor_guid_t t_op2 =
          require_only_key(op2_added.outputs, TensorSlotName::OUTPUT);

      ParallelLayerAddedResult op3_added =
          add_parallel_layer(pcg,
                             make_layer_attrs(relu_attrs),
                             {{TensorSlotName::INPUT, t_op1}},
                             {});

      ParallelLayerAddedResult op4_added = add_parallel_layer(
          /*pcg=*/pcg,
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
          /*=*/{});

      std::optional<SeriesParallelDecomposition> result =
          get_pcg_series_parallel_decomposition(pcg);
      std::optional<SeriesParallelDecomposition> correct = std::nullopt;

      CHECK(result == correct);
    }
  }
}
