#include "substitutions/pcg_pattern.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "substitutions/open_parallel_tensor_guid_t.h"
#include "substitutions/operator_pattern/operator_attribute_constraint.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "substitutions/tensor_pattern/tensor_attribute_pattern.h"
#include "utils/containers/get_only.h"
#include "utils/containers/require_only_key.h"
#include "utils/graph/instances/unordered_set_labelled_open_kwarg_dataflow_graph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("find_pattern_matches(PCGPattern, SubParallelComputationGraph)") {
    SUBCASE("simple case") {
      ParallelComputationGraphBuilder builder;

      positive_int batch_size = 16_p;
      int_ge_two batch_degree = 2_ge2;
      positive_int num_channels = 24_p;

      TensorShape a_shape = TensorShape{
          TensorDims{
              FFOrdered{
                  batch_size,
                  num_channels,
              },
          },
          DataType::FLOAT,
      };

      std::string a_name = "a";

      parallel_tensor_guid_t a_tensor = builder.create_input_tensor(a_shape);
      a_tensor =
          builder.parallel_partition(a_tensor, ff_dim_t{0_n}, batch_degree);

      positive_int outDim = 16_p;
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
      std::map<TensorSlotName, parallel_tensor_guid_t> x_incoming =
          get_incoming_tensors(pcg, x_matmul);
      REQUIRE(x_incoming.size() == 2);

      parallel_tensor_guid_t x_weights = x_incoming.at(TensorSlotName::WEIGHT);
      std::map<TensorSlotName, parallel_tensor_guid_t> y_incoming =
          get_incoming_tensors(pcg, y_matmul);
      REQUIRE(y_incoming.size() == 2);
      parallel_tensor_guid_t y_weights = y_incoming.at(TensorSlotName::WEIGHT);

      LabelledOpenKwargDataflowGraph<OperatorAttributePattern,
                                     TensorAttributePattern,
                                     int,
                                     TensorSlotName>
          g = LabelledOpenKwargDataflowGraph<OperatorAttributePattern,
                                             TensorAttributePattern,
                                             int,
                                             TensorSlotName>::
              create<UnorderedSetLabelledOpenKwargDataflowGraph<
                  OperatorAttributePattern,
                  TensorAttributePattern,
                  int,
                  TensorSlotName>>();

      TensorAttributePattern pattern_tensor_a =
          tensor_attribute_pattern_match_all();
      TensorAttributePattern pattern_tensor_b =
          tensor_attribute_pattern_match_all();
      TensorAttributePattern pattern_tensor_c =
          tensor_attribute_pattern_match_all();
      TensorAttributePattern pattern_tensor_x =
          tensor_attribute_pattern_match_all();
      TensorAttributePattern pattern_tensor_y =
          tensor_attribute_pattern_match_all();

      OperatorAttributePattern op_pattern_1 = OperatorAttributePattern{{
          op_type_equals_constraint(OperatorType::LINEAR),
      }};

      OperatorAttributePattern op_pattern_2 = op_pattern_1;

      KwargDataflowGraphInput<int> pt_a = g.add_input(0, pattern_tensor_a);
      KwargDataflowGraphInput<int> pt_b = g.add_input(1, pattern_tensor_b);
      KwargDataflowGraphInput<int> pt_c = g.add_input(2, pattern_tensor_c);

      KwargNodeAddedResult<TensorSlotName> op_pattern_1_added = g.add_node(
          /*node_label=*/op_pattern_1,
          /*inputs=*/
          {
              {
                  TensorSlotName::INPUT,
                  OpenKwargDataflowValue<int, TensorSlotName>{pt_a},
              },
              {
                  TensorSlotName::WEIGHT,
                  OpenKwargDataflowValue<int, TensorSlotName>{pt_b},
              },
          },
          /*output_labels=*/
          {
              {
                  TensorSlotName::OUTPUT,
                  pattern_tensor_x,
              },
          });
      PatternNode op_pattern_1_node = PatternNode{op_pattern_1_added.node};
      OpenKwargDataflowValue<int, TensorSlotName> pt_x =
          OpenKwargDataflowValue<int, TensorSlotName>{
              require_only_key(op_pattern_1_added.outputs,
                               TensorSlotName::OUTPUT),
          };

      KwargNodeAddedResult<TensorSlotName> op_pattern_2_added = g.add_node(
          /*node_label=*/op_pattern_2,
          /*inputs=*/
          {
              {
                  TensorSlotName::INPUT,
                  OpenKwargDataflowValue<int, TensorSlotName>{pt_a},
              },
              {
                  TensorSlotName::WEIGHT,
                  OpenKwargDataflowValue<int, TensorSlotName>{pt_c},
              },
          },
          /*outputs_labels=*/
          {
              {
                  TensorSlotName::OUTPUT,
                  pattern_tensor_y,
              },
          });
      PatternNode op_pattern_2_node = PatternNode{op_pattern_2_added.node};
      OpenKwargDataflowValue<int, TensorSlotName> pt_y =
          OpenKwargDataflowValue<int, TensorSlotName>{
              require_only_key(op_pattern_2_added.outputs,
                               TensorSlotName::OUTPUT),
          };

      PCGPattern pattern = PCGPattern{g};

      std::set<PCGPatternMatch> result =
          set_of(find_pattern_matches(pattern, sub_pcg_from_full_pcg(pcg)));

      PCGPatternMatch match1 = PCGPatternMatch{
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

      PCGPatternMatch match2 = PCGPatternMatch{
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

      std::set<PCGPatternMatch> correct = {match1, match2};

      CHECK(result == correct);
    }

    SUBCASE("pcg is a chain") {
      ParallelComputationGraphBuilder builder;

      positive_int batch_size = 16_p;
      int_ge_two batch_degree = 2_ge2;
      positive_int num_channels = 24_p;

      TensorShape a_shape = TensorShape{
          TensorDims{
              FFOrdered{
                  batch_size,
                  num_channels,
              },
          },
          DataType::FLOAT,
      };

      std::string a_name = "a";

      parallel_tensor_guid_t a_tensor = builder.create_input_tensor(a_shape);
      a_tensor =
          builder.parallel_partition(a_tensor, ff_dim_t{0_n}, batch_degree);

      positive_int outDim = 16_p;
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
          builder.dense(t0,
                        outDim,
                        /*activation=*/std::nullopt,
                        /*use_bias=*/false,
                        DataType::FLOAT,
                        /*kernel_initializer=*/std::nullopt,
                        /*bias_initializer=*/std::nullopt,
                        y_matmul_name);
      parallel_tensor_guid_t t2 =
          builder.dense(t1,
                        outDim,
                        /*activation=*/std::nullopt,
                        /*use_bias=*/false,
                        DataType::FLOAT,
                        /*kernel_initializer=*/std::nullopt,
                        /*bias_initializer=*/std::nullopt);
      parallel_tensor_guid_t t3 =
          builder.dense(t2,
                        outDim,
                        /*activation=*/std::nullopt,
                        /*use_bias=*/false,
                        DataType::FLOAT,
                        /*kernel_initializer=*/std::nullopt,
                        /*bias_initializer=*/std::nullopt);
      ParallelComputationGraph pcg = builder.pcg;

      LabelledOpenKwargDataflowGraph<OperatorAttributePattern,
                                     TensorAttributePattern,
                                     int,
                                     TensorSlotName>
          g = LabelledOpenKwargDataflowGraph<OperatorAttributePattern,
                                             TensorAttributePattern,
                                             int,
                                             TensorSlotName>::
              create<UnorderedSetLabelledOpenKwargDataflowGraph<
                  OperatorAttributePattern,
                  TensorAttributePattern,
                  int,
                  TensorSlotName>>();

      TensorAttributePattern pattern_tensor_a =
          tensor_attribute_pattern_match_all();
      TensorAttributePattern pattern_tensor_b =
          tensor_attribute_pattern_match_all();
      TensorAttributePattern pattern_tensor_c =
          tensor_attribute_pattern_match_all();
      TensorAttributePattern pattern_tensor_x =
          tensor_attribute_pattern_match_all();
      TensorAttributePattern pattern_tensor_y =
          tensor_attribute_pattern_match_all();

      OperatorAttributePattern op_pattern_1 = OperatorAttributePattern{{
          op_type_equals_constraint(OperatorType::LINEAR),
      }};

      OperatorAttributePattern op_pattern_2 = op_pattern_1;

      KwargDataflowGraphInput<int> pt_a = g.add_input(0, pattern_tensor_a);
      KwargDataflowGraphInput<int> pt_b = g.add_input(1, pattern_tensor_b);
      KwargDataflowGraphInput<int> pt_c = g.add_input(2, pattern_tensor_c);

      KwargNodeAddedResult<TensorSlotName> op_pattern_1_added = g.add_node(
          /*node_label=*/op_pattern_1,
          /*inputs=*/
          {
              {
                  TensorSlotName::INPUT,
                  OpenKwargDataflowValue<int, TensorSlotName>{pt_a},
              },
              {
                  TensorSlotName::WEIGHT,
                  OpenKwargDataflowValue<int, TensorSlotName>{pt_b},
              },
          },
          /*output_labels=*/
          {
              {
                  TensorSlotName::OUTPUT,
                  pattern_tensor_x,
              },
          });
      PatternNode op_pattern_1_node = PatternNode{op_pattern_1_added.node};
      OpenKwargDataflowValue<int, TensorSlotName> pt_x =
          OpenKwargDataflowValue<int, TensorSlotName>{
              require_only_key(op_pattern_1_added.outputs,
                               TensorSlotName::OUTPUT),
          };

      KwargNodeAddedResult<TensorSlotName> op_pattern_2_added = g.add_node(
          /*node_label=*/op_pattern_2,
          /*inputs=*/
          {
              {
                  TensorSlotName::INPUT,
                  OpenKwargDataflowValue<int, TensorSlotName>{pt_x},
              },
              {
                  TensorSlotName::WEIGHT,
                  OpenKwargDataflowValue<int, TensorSlotName>{pt_c},
              },
          },
          /*output_labels=*/
          {
              {
                  TensorSlotName::OUTPUT,
                  pattern_tensor_y,
              },
          });
      PatternNode op_pattern_2_node = PatternNode{op_pattern_2_added.node};

      PCGPattern pattern = PCGPattern{g};

      std::set<PCGPatternMatch> result =
          set_of(find_pattern_matches(pattern, sub_pcg_from_full_pcg(pcg)));

      CHECK(result.size() == 3);
    }
  }
}
