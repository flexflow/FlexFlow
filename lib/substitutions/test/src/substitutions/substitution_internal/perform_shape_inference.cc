#include <doctest/doctest.h>
#include "substitutions/substitution_internal/perform_shape_inference.h"
#include "utils/containers/get_only.h"
#include "utils/graph/instances/unordered_set_labelled_open_dataflow_graph.h"
#include "utils/graph/labelled_open_dataflow_graph/labelled_open_dataflow_graph.h"
#include "utils/integer_conversions.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/get_graph_data.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("perform_shape_inference") {
    auto g = LabelledOpenDataflowGraph<ParallelLayerAttrs, std::monostate>::create<
            UnorderedSetLabelledOpenDataflowGraph<ParallelLayerAttrs, std::monostate>>();

    int in_channels = 24;
    int out_channels = 16;
    int batch_size = 4;
    int batch_degree = 2;

    DataflowGraphInput i0 = g.add_input({});
    ParallelTensorShape i0_shape = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{size_t_from_int(batch_size), batch_degree},
                ShardParallelDim{size_t_from_int(in_channels), 1},
            },
            ReplicaParallelDimSet{
                SumDegree{1},
                DiscardCopyDegree{1},
            },
        },
        DataType::FLOAT,
    };

    bool use_bias = false;
    LinearAttrs n1_op_attrs = LinearAttrs{
        out_channels,
        use_bias,
        DataType::FLOAT,
        std::nullopt,
        std::nullopt,
    };
    ParallelLayerAttrs n1_attrs = ParallelLayerAttrs{
        PCGOperatorAttrs{
            n1_op_attrs,
        },
        std::nullopt,
    };

    ElementUnaryAttrs n2_op_attrs = ElementUnaryAttrs{
        OperatorType::RELU,
        std::nullopt,
    };
    ParallelLayerAttrs n2_attrs = ParallelLayerAttrs{
        PCGOperatorAttrs{
            n2_op_attrs,
        },
        std::nullopt,
    };

    ParallelTensorShape n1_output_shape =
        throw_if_unexpected(get_output_shape(n1_op_attrs, i0_shape));
    ParallelTensorShape n1_weight_shape =
        throw_if_unexpected(get_kernel_shape(n1_op_attrs, i0_shape));
    ParallelTensorShape n2_output_shape =
        throw_if_unexpected(get_output_shape(n2_op_attrs, n1_output_shape));

    ParallelLayerAttrs n1_weight_attrs = ParallelLayerAttrs{
        PCGOperatorAttrs{
            WeightAttrs{get_reduced_shape(n1_weight_shape)},
        },
        std::nullopt,
    };

    ParallelLayerAttrs n1_weight_replicate_attrs = ParallelLayerAttrs{
        PCGOperatorAttrs{
            ReplicateAttrs{batch_degree},
        },
        std::nullopt,
    };

    NodeAddedResult n1_weight_added_result =
        g.add_node(n1_weight_attrs, {}, {{}});
    Node n1_weight_node = n1_weight_added_result.node;
    DataflowOutput n1_weight = get_only(n1_weight_added_result.outputs);

    NodeAddedResult n1_weight_replicate_added_result = g.add_node(
        n1_weight_replicate_attrs, {OpenDataflowValue{n1_weight}}, {{}});
    Node n1_weight_replicate_node = n1_weight_replicate_added_result.node;
    DataflowOutput n1_weight_replicated =
        get_only(n1_weight_replicate_added_result.outputs);

    NodeAddedResult n1_added_result = g.add_node(
        n1_attrs,
        {OpenDataflowValue{i0}, OpenDataflowValue{n1_weight_replicated}},
        {{}});
    Node n1 = n1_added_result.node;
    DataflowOutput o1 = get_only(n1_added_result.outputs);

    NodeAddedResult n2_added_result =
        g.add_node(n2_attrs, {OpenDataflowValue{o1}}, {{}});
    Node n2 = n2_added_result.node;
    DataflowOutput o2 = get_only(n2_added_result.outputs);

    std::unordered_map<DataflowGraphInput, ParallelTensorShape> input_shapes = {
        {i0, i0_shape},
    };

    LabelledOpenDataflowGraphView<ParallelLayerAttrs, ParallelTensorShape>
        result = perform_shape_inference(g, input_shapes);

    LabelledOpenDataflowGraphData<ParallelLayerAttrs, ParallelTensorShape>
        result_data = get_graph_data(result);

    LabelledOpenDataflowGraphData<ParallelLayerAttrs, ParallelTensorShape>
        correct_data = LabelledOpenDataflowGraphData<ParallelLayerAttrs,
                                                     ParallelTensorShape>{
            {
                {n1, n1_attrs},
                {n2, n2_attrs},
                {n1_weight_node, n1_weight_attrs},
                {n1_weight_replicate_node, n1_weight_replicate_attrs},
            },
            {
                OpenDataflowEdge{
                    DataflowInputEdge{
                        i0,
                        DataflowInput{n1, 0},
                    },
                },
                OpenDataflowEdge{DataflowEdge{
                    DataflowOutput{n1_weight_node, 0},
                    DataflowInput{n1_weight_replicate_node, 0},
                }},
                OpenDataflowEdge{
                    DataflowEdge{
                        DataflowOutput{n1_weight_replicate_node, 0},
                        DataflowInput{n1, 1},
                    },
                },
                OpenDataflowEdge{DataflowEdge{
                    DataflowOutput{n1, 0},
                    DataflowInput{n2, 0},
                }},
            },
            {i0},
            {{
                 OpenDataflowValue{i0},
                 i0_shape,
             },
             {
                 OpenDataflowValue{DataflowOutput{n1_weight_node, 0}},
                 lift_to_parallel(get_reduced_shape(n1_weight_shape)),
             },
             {
                 OpenDataflowValue{DataflowOutput{n1_weight_replicate_node, 0}},
                 n1_weight_shape,
             },
             {
                 OpenDataflowValue{DataflowOutput{n1, 0}},
                 n1_output_shape,
             },
             {
                 OpenDataflowValue{DataflowOutput{n2, 0}},
                 n2_output_shape,
             }}};

    CHECK(result_data == correct_data);
  }
}
