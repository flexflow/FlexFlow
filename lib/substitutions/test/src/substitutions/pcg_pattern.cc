#include "substitutions/pcg_pattern.h"
#include "substitutions/open_parallel_tensor_guid_t.h"
#include "substitutions/operator_pattern/operator_attribute_constraint.h"
#include "substitutions/tensor_pattern/tensor_attribute_pattern.h"
#include "utils/containers/get_only.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "test/utils/doctest.h"
#include "utils/graph/instances/unordered_set_labelled_open_dataflow_graph.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("find_pattern_matches(PCGPattern, SubParallelComputationGraph)") {
    ParallelComputationGraphBuilder builder;

    size_t batch_size = 16;
    int batch_degree = 2;
    size_t num_channels = 24;

    ParallelTensorShape a_shape = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{batch_size, batch_degree},
                ShardParallelDim{num_channels, 1},
            },
            ReplicaParallelDimSet{
                SumDegree{1},
                DiscardCopyDegree{1},
            },
        },
        DataType::FLOAT,
    };
    std::string a_name = "a";

    parallel_tensor_guid_t a_tensor =
        builder.create_input_tensor(a_shape, /*create_grad=*/true, a_name);

    int outDim = 16;
    std::string x_matmul_name = "x_matmul";
    std::string y_matmul_name = "y_matmul";
    parallel_tensor_guid_t t0 =
        builder.dense(a_tensor,
                      outDim,
                      /*activation=*/std::nullopt,
                      /*use_bias=*/false,
                      DataType::FLOAT,
                      /*kernel_initializer=*/std::nullopt,
                      /*bias_initializer=*/std::nullopt,
                      x_matmul_name);
    parallel_tensor_guid_t t1 =
        builder.dense(a_tensor,
                      outDim,
                      /*activation=*/std::nullopt,
                      /*use_bias=*/false,
                      DataType::FLOAT,
                      /*kernel_initializer=*/std::nullopt,
                      /*bias_initializer=*/std::nullopt,
                      y_matmul_name);
    parallel_tensor_guid_t t2 = builder.add(t0, t1);

    ParallelComputationGraph pcg = builder.pcg;
    parallel_layer_guid_t x_matmul =
        get_parallel_layer_by_name(pcg, x_matmul_name);
    parallel_layer_guid_t y_matmul =
        get_parallel_layer_by_name(pcg, y_matmul_name);
    std::vector<parallel_tensor_guid_t> x_inputs =
        get_layer_inputs(pcg, x_matmul);
    REQUIRE(x_inputs.size() == 2);
    parallel_tensor_guid_t x_weights = x_inputs.at(1);
    std::vector<parallel_tensor_guid_t> y_inputs =
        get_layer_inputs(pcg, y_matmul);
    REQUIRE(y_inputs.size() == 2);
    parallel_tensor_guid_t y_weights = y_inputs.at(1);

    LabelledOpenDataflowGraph<OperatorAttributePattern, TensorAttributePattern>
        g = LabelledOpenDataflowGraph<OperatorAttributePattern,
                                      TensorAttributePattern>::
            create<UnorderedSetLabelledOpenDataflowGraph<
                OperatorAttributePattern,
                TensorAttributePattern>>();

    TensorAttributePattern pattern_tensor_a = tensor_attribute_pattern_match_all();
    TensorAttributePattern pattern_tensor_b = tensor_attribute_pattern_match_all();
    TensorAttributePattern pattern_tensor_c = tensor_attribute_pattern_match_all();
    TensorAttributePattern pattern_tensor_x = tensor_attribute_pattern_match_all();
    TensorAttributePattern pattern_tensor_y = tensor_attribute_pattern_match_all();

    OperatorAttributePattern op_pattern_1 =
        OperatorAttributePattern{{
          op_type_equals_constraint(OperatorType::LINEAR),
        }};

    OperatorAttributePattern op_pattern_2 = op_pattern_1;

    DataflowGraphInput pt_a = g.add_input(pattern_tensor_a);
    DataflowGraphInput pt_b = g.add_input(pattern_tensor_b);
    DataflowGraphInput pt_c = g.add_input(pattern_tensor_c);

    NodeAddedResult op_pattern_1_added =
        g.add_node(op_pattern_1,
                   {OpenDataflowValue{pt_a}, OpenDataflowValue{pt_b}},
                   {pattern_tensor_x});
    PatternNode op_pattern_1_node = PatternNode{op_pattern_1_added.node};
    OpenDataflowValue pt_x =
        OpenDataflowValue{get_only(op_pattern_1_added.outputs)};

    NodeAddedResult op_pattern_2_added =
        g.add_node(op_pattern_2,
                   {OpenDataflowValue{pt_a}, OpenDataflowValue{pt_c}},
                   {pattern_tensor_y});
    PatternNode op_pattern_2_node = PatternNode{op_pattern_2_added.node};
    OpenDataflowValue pt_y =
        OpenDataflowValue{get_only(op_pattern_2_added.outputs)};

    PCGPattern pattern = PCGPattern{g};

    std::unordered_set<PCGPatternMatch> result =
        unordered_set_of(
            find_pattern_matches(pattern, sub_pcg_from_full_pcg(pcg)));

    PCGPatternMatch match1 =
        PCGPatternMatch{
            bidict<PatternNode, parallel_layer_guid_t>{
                {op_pattern_1_node, x_matmul},
                {op_pattern_2_node, y_matmul},
            },
            bidict<PatternInput, open_parallel_tensor_guid_t>{
                {PatternInput{pt_a},
                  open_parallel_tensor_guid_from_closed(a_tensor)},
                {PatternInput{pt_b},
                  open_parallel_tensor_guid_from_closed(x_weights)},
                {PatternInput{pt_c},
                  open_parallel_tensor_guid_from_closed(y_weights)},
            }};

    PCGPatternMatch match2 =
        PCGPatternMatch{
            bidict<PatternNode, parallel_layer_guid_t>{
                {op_pattern_1_node, y_matmul},
                {op_pattern_2_node, x_matmul},
            },
            bidict<PatternInput, open_parallel_tensor_guid_t>{
                {PatternInput{pt_a},
                  open_parallel_tensor_guid_from_closed(a_tensor)},
                {PatternInput{pt_b},
                  open_parallel_tensor_guid_from_closed(y_weights)},
                {PatternInput{pt_c},
                  open_parallel_tensor_guid_from_closed(x_weights)},
            }};

    std::unordered_set<PCGPatternMatch> correct = {match1, match2};

    CHECK(result == correct);
  }
}
