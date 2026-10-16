#include "substitutions/apply_substitution/perform_shape_inference.h"
#include "op-attrs/ops/element_unary.h"
#include "op-attrs/ops/linear.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/containers/get_only.h"
#include "utils/containers/require_only_key.h"
#include "utils/graph/instances/unordered_set_labelled_open_kwarg_dataflow_graph.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/get_labelled_open_kwarg_dataflow_graph_data.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/labelled_open_kwarg_dataflow_graph_data.dtg.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/labelled_open_kwarg_dataflow_graph.h"
#include "utils/integer_conversions.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("perform_shape_inference") {
    auto g = LabelledOpenKwargDataflowGraph<ParallelLayerAttrs,
                                            std::monostate,
                                            int,
                                            TensorSlotName>::
        create<UnorderedSetLabelledOpenKwargDataflowGraph<ParallelLayerAttrs,
                                                          std::monostate,
                                                          int,
                                                          TensorSlotName>>();

    positive_int in_channels = 24_p;
    positive_int out_channels = 16_p;
    positive_int batch_size = 4_p;
    positive_int batch_degree = 2_p;

    KwargDataflowGraphInput<int> i0 = g.add_input(0, std::monostate{});
    ParallelTensorShape i0_shape = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{batch_size, batch_degree},
                ShardParallelDim{in_channels, 1_p},
            },
            ReplicaParallelDimSet{
                SumDegree{1_p},
                DiscardCopyDegree{1_p},
            },
        },
        DataType::FLOAT,
    };

    bool use_bias = false;
    LinearAttrs n1_op_attrs = LinearAttrs{
        /*out_channels=*/out_channels,
        /*use_bias=*/use_bias,
        /*data_type=*/DataType::FLOAT,
        /*activation=*/std::nullopt,
        /*regularizer=*/std::nullopt,
    };
    ParallelLayerAttrs n1_attrs = ParallelLayerAttrs{
        /*op_attrs=*/PCGOperatorAttrs{
            n1_op_attrs,
        },
        /*name=*/std::nullopt,
    };

    ElementUnaryAttrs n2_op_attrs = ElementUnaryAttrs{
        /*op_type=*/ElementUnaryOp::RELU,
        /*scalar=*/std::nullopt,
    };
    ParallelLayerAttrs n2_attrs = ParallelLayerAttrs{
        /*op_attrs=*/PCGOperatorAttrs{
            n2_op_attrs,
        },
        /*name=*/std::nullopt,
    };

    ParallelTensorShape n1_output_shape =
        linear_get_output_parallel_shape(n1_op_attrs, i0_shape);
    ParallelTensorShape n1_weight_shape =
        linear_get_projection_parallel_shape(n1_op_attrs, i0_shape);
    ParallelTensorShape n2_output_shape =
        element_unary_get_output_parallel_shape(n2_op_attrs, n1_output_shape);

    ParallelLayerAttrs n1_weight_attrs = ParallelLayerAttrs{
        PCGOperatorAttrs{
            WeightAttrs{
                get_reduced_shape(n1_weight_shape),
                InitializerAttrs{ZeroInitializerAttrs{}},
            },
        },
        std::nullopt,
    };

    ParallelLayerAttrs n1_weight_replicate_attrs = ParallelLayerAttrs{
        PCGOperatorAttrs{
            ReplicateAttrs{int_ge_two{batch_degree}},
        },
        std::nullopt,
    };

    KwargNodeAddedResult<TensorSlotName> n1_weight_added_result = g.add_node(
        /*node_labels=*/n1_weight_attrs,
        /*inputs=*/{},
        /*output_labels=*/
        {
            {
                TensorSlotName::OUTPUT,
                {},
            },
        });
    Node n1_weight_node = n1_weight_added_result.node;
    KwargDataflowOutput<TensorSlotName> n1_weight = require_only_key(
        n1_weight_added_result.outputs, TensorSlotName::OUTPUT);

    KwargNodeAddedResult<TensorSlotName> n1_weight_replicate_added_result =
        g.add_node(
            /*node_label=*/n1_weight_replicate_attrs,
            /*inputs=*/
            {
                {
                    TensorSlotName::INPUT,
                    OpenKwargDataflowValue<int, TensorSlotName>{n1_weight},
                },
            },
            /*outupt_labels=*/
            {
                {
                    TensorSlotName::OUTPUT,
                    std::monostate{},
                },
            });
    Node n1_weight_replicate_node = n1_weight_replicate_added_result.node;
    KwargDataflowOutput<TensorSlotName> n1_weight_replicated = require_only_key(
        n1_weight_replicate_added_result.outputs, TensorSlotName::OUTPUT);

    KwargNodeAddedResult<TensorSlotName> n1_added_result = g.add_node(
        /*node_label=*/n1_attrs,
        /*inputs=*/
        {
            {
                TensorSlotName::INPUT,
                OpenKwargDataflowValue<int, TensorSlotName>{i0},
            },
            {
                TensorSlotName::WEIGHT,
                OpenKwargDataflowValue<int, TensorSlotName>{
                    n1_weight_replicated},
            },
        },
        /*output_labels=*/
        {
            {
                TensorSlotName::OUTPUT,
                std::monostate{},
            },
        });
    Node n1 = n1_added_result.node;
    KwargDataflowOutput<TensorSlotName> o1 =
        require_only_key(n1_added_result.outputs, TensorSlotName::OUTPUT);

    KwargNodeAddedResult<TensorSlotName> n2_added_result = g.add_node(
        /*node_labels=*/n2_attrs,
        /*inputs=*/
        {
            {
                TensorSlotName::INPUT,
                OpenKwargDataflowValue<int, TensorSlotName>{o1},
            },
        },
        /*output_labels=*/
        {
            {
                TensorSlotName::OUTPUT,
                {},
            },
        });
    Node n2 = n2_added_result.node;
    KwargDataflowOutput<TensorSlotName> o2 =
        require_only_key(n2_added_result.outputs, TensorSlotName::OUTPUT);

    std::map<KwargDataflowGraphInput<int>, ParallelTensorShape> input_shapes = {
        {i0, i0_shape},
    };

    LabelledOpenKwargDataflowGraphView<ParallelLayerAttrs,
                                       ParallelTensorShape,
                                       int,
                                       TensorSlotName>
        result = perform_shape_inference(g, input_shapes);

    LabelledOpenKwargDataflowGraphData<ParallelLayerAttrs,
                                       ParallelTensorShape,
                                       int,
                                       TensorSlotName>
        result_data = get_labelled_open_kwarg_dataflow_graph_data(result);

    LabelledOpenKwargDataflowGraphData<ParallelLayerAttrs,
                                       ParallelTensorShape,
                                       int,
                                       TensorSlotName>
        correct_data = LabelledOpenKwargDataflowGraphData<ParallelLayerAttrs,
                                                          ParallelTensorShape,
                                                          int,
                                                          TensorSlotName>{
            {
                {n1, n1_attrs},
                {n2, n2_attrs},
                {n1_weight_node, n1_weight_attrs},
                {n1_weight_replicate_node, n1_weight_replicate_attrs},
            },
            {
                OpenKwargDataflowEdge<int, TensorSlotName>{
                    KwargDataflowInputEdge<int, TensorSlotName>{
                        i0,
                        KwargDataflowInput{
                            n1,
                            TensorSlotName::INPUT,
                        },
                    },
                },
                OpenKwargDataflowEdge<int, TensorSlotName>{
                    KwargDataflowEdge<TensorSlotName>{
                        KwargDataflowOutput{
                            n1_weight_node,
                            TensorSlotName::OUTPUT,
                        },
                        KwargDataflowInput{
                            n1_weight_replicate_node,
                            TensorSlotName::INPUT,
                        },
                    },
                },
                OpenKwargDataflowEdge<int, TensorSlotName>{
                    KwargDataflowEdge<TensorSlotName>{
                        KwargDataflowOutput{
                            n1_weight_replicate_node,
                            TensorSlotName::OUTPUT,
                        },
                        KwargDataflowInput{
                            n1,
                            TensorSlotName::WEIGHT,
                        },
                    },
                },
                OpenKwargDataflowEdge<int, TensorSlotName>{
                    KwargDataflowEdge<TensorSlotName>{
                        KwargDataflowOutput{
                            n1,
                            TensorSlotName::OUTPUT,
                        },
                        KwargDataflowInput{
                            n2,
                            TensorSlotName::INPUT,
                        },
                    },
                },
            },
            {i0},
            {
                {
                    OpenKwargDataflowValue<int, TensorSlotName>{i0},
                    i0_shape,
                },
                {
                    OpenKwargDataflowValue<int, TensorSlotName>{
                        KwargDataflowOutput<TensorSlotName>{
                            n1_weight_node,
                            TensorSlotName::OUTPUT,
                        },
                    },
                    lift_shape_to_parallel(get_reduced_shape(n1_weight_shape)),
                },
                {
                    OpenKwargDataflowValue<int, TensorSlotName>{
                        KwargDataflowOutput<TensorSlotName>{
                            n1_weight_replicate_node,
                            TensorSlotName::OUTPUT,
                        }},
                    n1_weight_shape,
                },
                {
                    OpenKwargDataflowValue<int, TensorSlotName>{
                        KwargDataflowOutput<TensorSlotName>{
                            n1,
                            TensorSlotName::OUTPUT,
                        },
                    },
                    n1_output_shape,
                },
                {
                    OpenKwargDataflowValue<int, TensorSlotName>{
                        KwargDataflowOutput<TensorSlotName>{
                            n2,
                            TensorSlotName::OUTPUT,
                        },
                    },
                    n2_output_shape,
                },
            }};

    CHECK(result_data == correct_data);
  }
}
