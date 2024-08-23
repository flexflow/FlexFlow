#include <doctest/doctest.h>
#include "substitutions/open_parallel_tensor_guid_t.h"
#include "substitutions/operator_pattern/operator_attribute_constraint.h"
#include "substitutions/output_graph/output_operator_attrs_assignment.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "substitutions/substitution.h"
#include "substitutions/pcg_pattern_builder.h"
#include "substitutions/tensor_pattern/tensor_attribute_pattern.h"
#include "utils/containers/get_only.h"
#include "utils/graph/instances/unordered_set_labelled_open_dataflow_graph.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/get_graph_data.h"
#include "utils/integer_conversions.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  // TEST_CASE("is_valid_substitution") {
  //   FAIL("TODO");
  // }

  TEST_CASE("perform_shape_inference") {
    LabelledOpenDataflowGraph<ParallelLayerAttrs, std::monostate> g = 
      LabelledOpenDataflowGraph<ParallelLayerAttrs, std::monostate>::create<
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
    LinearAttrs n1_op_attrs = 
      LinearAttrs{
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

    ElementUnaryAttrs n2_op_attrs = 
        ElementUnaryAttrs{
          OperatorType::RELU,
          std::nullopt,
        };
    ParallelLayerAttrs n2_attrs = ParallelLayerAttrs{
      PCGOperatorAttrs{
        n2_op_attrs,
      },
      std::nullopt,
    };

    ParallelTensorShape n1_output_shape = throw_if_unexpected(get_output_shape(n1_op_attrs, i0_shape));
    ParallelTensorShape n1_weight_shape = throw_if_unexpected(get_kernel_shape(n1_op_attrs, i0_shape));
    ParallelTensorShape n2_output_shape = throw_if_unexpected(get_output_shape(n2_op_attrs, n1_output_shape));


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

    NodeAddedResult n1_weight_added_result = g.add_node(n1_weight_attrs, {}, {{}});
    Node n1_weight_node = n1_weight_added_result.node;
    DataflowOutput n1_weight = get_only(n1_weight_added_result.outputs);

    NodeAddedResult n1_weight_replicate_added_result = g.add_node(n1_weight_replicate_attrs, {OpenDataflowValue{n1_weight}}, {{}});
    Node n1_weight_replicate_node = n1_weight_replicate_added_result.node;
    DataflowOutput n1_weight_replicated = get_only(n1_weight_replicate_added_result.outputs);

    NodeAddedResult n1_added_result = g.add_node(n1_attrs, {OpenDataflowValue{i0}, OpenDataflowValue{n1_weight_replicated}}, {{}});
    Node n1 = n1_added_result.node;
    DataflowOutput o1 = get_only(n1_added_result.outputs);

    NodeAddedResult n2_added_result = g.add_node(n2_attrs, {OpenDataflowValue{o1}}, {{}});
    Node n2 = n2_added_result.node;
    DataflowOutput o2 = get_only(n2_added_result.outputs);

    std::unordered_map<DataflowGraphInput, ParallelTensorShape> input_shapes = {
      {i0, i0_shape},
    };

    LabelledOpenDataflowGraphView<ParallelLayerAttrs, ParallelTensorShape> result = 
      perform_shape_inference(g, input_shapes);

    LabelledOpenDataflowGraphData<ParallelLayerAttrs, ParallelTensorShape> result_data = get_graph_data(result);

    
    LabelledOpenDataflowGraphData<ParallelLayerAttrs, ParallelTensorShape> correct_data = 
      LabelledOpenDataflowGraphData<ParallelLayerAttrs, ParallelTensorShape>{
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
              DataflowInput{
                n1, 0
              },
            },
          },
          OpenDataflowEdge{
            DataflowEdge{
              DataflowOutput{
                n1_weight_node, 0
              },
              DataflowInput{
                n1_weight_replicate_node, 0
              },
            }
          },
          OpenDataflowEdge{
            DataflowEdge{
              DataflowOutput{
                n1_weight_replicate_node, 0
              },
              DataflowInput{
                n1, 1
              },
            },
          },
          OpenDataflowEdge{
            DataflowEdge{
              DataflowOutput{
                n1, 0
              },
              DataflowInput{
                n2, 0
              },
            }
          },
        },
        { i0 },
        {
          {
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
          }
        }
      };
    
    CHECK(result_data == correct_data);
  }

  TEST_CASE("evaluate_substitution_output") {
    auto pattern_g = 
      LabelledOpenDataflowGraph<OperatorAttributePattern, TensorAttributePattern>::create<
        UnorderedSetLabelledOpenDataflowGraph<OperatorAttributePattern, TensorAttributePattern>>();

    PatternInput pattern_i_activation = PatternInput{pattern_g.add_input(tensor_attribute_pattern_match_all())};
    PatternInput pattern_i_weights = PatternInput{pattern_g.add_input(tensor_attribute_pattern_match_all())};

    OperatorAttributePattern mm_pattern = OperatorAttributePattern{{
      op_type_equals_constraint(OperatorType::LINEAR),
      op_attr_key_equals(OperatorAttributeKey::ACTIVATION,
                         OperatorAttributeValue{std::optional<Activation>{std::nullopt}}),
    }};
    NodeAddedResult mm_added = pattern_g.add_node(mm_pattern,
                                    {OpenDataflowValue{pattern_i_activation.raw_dataflow_graph_input}, 
                                     OpenDataflowValue{pattern_i_weights.raw_dataflow_graph_input}},
                                    {tensor_attribute_pattern_match_all()});
    PatternNode pattern_mm_node = PatternNode{mm_added.node};
    DataflowOutput mm_output = get_only(mm_added.outputs);

    OperatorAttributePattern relu_pattern = OperatorAttributePattern{{
      op_type_equals_constraint(OperatorType::RELU),
    }};
    NodeAddedResult relu_added = pattern_g.add_node(relu_pattern,
                                            {OpenDataflowValue{mm_output}},
                                            {tensor_attribute_pattern_match_all()});
    PatternNode pattern_relu_node = PatternNode{relu_added.node};
    DataflowOutput relu_output = get_only(relu_added.outputs);

    LabelledOpenDataflowGraph<OutputOperatorAttrsAssignment, std::monostate> output_g = 
      LabelledOpenDataflowGraph<OutputOperatorAttrsAssignment, std::monostate>::create<
        UnorderedSetLabelledOpenDataflowGraph<OutputOperatorAttrsAssignment, std::monostate>>();

    OutputGraphExprInput output_i_activation = OutputGraphExprInput{output_g.add_input({})};
    OutputGraphExprInput output_i_weights = OutputGraphExprInput{output_g.add_input({})};

    OutputOperatorAttrsAssignment fused_mm_relu_attrs_assignment = OutputOperatorAttrsAssignment{{
      set_attr_to_constant(OperatorAttributeKey::OP_TYPE, OperatorAttributeValue{OperatorType::LINEAR}),
      copy_attr_from_pattern_node(OperatorAttributeKey::OUT_CHANNELS, pattern_mm_node),
      copy_attr_from_pattern_node(OperatorAttributeKey::USE_BIAS, pattern_mm_node),
      copy_attr_from_pattern_node(OperatorAttributeKey::DATA_TYPE, pattern_mm_node),
      set_attr_to_constant(OperatorAttributeKey::ACTIVATION, OperatorAttributeValue{Activation::RELU}),
      copy_attr_from_pattern_node(OperatorAttributeKey::REGULARIZER, pattern_mm_node),
    }};
    NodeAddedResult fused_mm_relu_added = output_g.add_node(fused_mm_relu_attrs_assignment, 
                                          {OpenDataflowValue{output_i_activation.raw_dataflow_graph_input}, 
                                           OpenDataflowValue{output_i_weights.raw_dataflow_graph_input}},
                                          {{}});
    OutputGraphExprNode fused_mm_relu_node = OutputGraphExprNode{fused_mm_relu_added.node};
    DataflowOutput fused_mm_relu_output = get_only(fused_mm_relu_added.outputs);
    
    Substitution sub = Substitution{
      PCGPattern{pattern_g},
      OutputGraphExpr{output_g},
      bidict<PatternInput, OutputGraphExprInput>{
        {
          pattern_i_activation,
          output_i_activation,
        },
        {
          pattern_i_weights,
          output_i_weights,
        },
      },
      bidict<PatternNodeOutput, OutputGraphExprNodeOutput>{
        {
          PatternNodeOutput{relu_output},
          OutputGraphExprNodeOutput{fused_mm_relu_output},
        },
      },
    };

    int in_channels = 24;
    int batch_size = 4;
    int batch_degree = 2;
    std::string mm_match = "mm_match";
    std::string relu_match = "relu_match";

    SubParallelComputationGraph pcg = [&] {
      ParallelComputationGraphBuilder b;
      parallel_tensor_guid_t t = b.create_input_tensor(
        ParallelTensorShape{
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
        }
      );
      t = b.dense(t, 
                  /*outDim=*/16,
                  /*activation=*/std::nullopt);
      t = b.gelu(t);
      t = b.dense(t,
                  /*outDim=*/12,
                  /*activation=*/std::nullopt,
                  /*use_bias=*/false,
                  /*data_type=*/DataType::FLOAT,
                  /*kernel_initializer=*/std::nullopt,
                  /*bias_initializer=*/std::nullopt,
                  /*name=*/mm_match);
      t = b.relu(t, 
                 /*name=*/relu_match);
      t = b.dense(t, 
                  /*outDim=*/8,
                  /*activation=*/Activation::RELU);
    
      return sub_pcg_from_full_pcg(b.pcg);
    }();

    parallel_layer_guid_t mm_match_layer = get_parallel_layer_by_name(pcg, mm_match);
    parallel_layer_guid_t relu_match_layer = get_parallel_layer_by_name(pcg, relu_match);
    open_parallel_tensor_guid_t mm_match_layer_input_activations = get_layer_inputs(pcg, mm_match_layer).at(0);
    open_parallel_tensor_guid_t mm_match_layer_input_weights = get_layer_inputs(pcg, mm_match_layer).at(1);

    PCGPatternMatch match = PCGPatternMatch{
      bidict<PatternNode, parallel_layer_guid_t>{
        { pattern_mm_node, mm_match_layer },
        { pattern_relu_node, relu_match_layer },
      },
      std::unordered_map<PatternInput, open_parallel_tensor_guid_t>{
        {
          PatternInput{pattern_i_activation}, 
          mm_match_layer_input_activations,
        },
        {
          PatternInput{pattern_i_weights},
          mm_match_layer_input_weights,
        }
      },
    };

    
    SUBCASE("evaluate_substitution_output") {
      std::pair<
        SubParallelComputationGraph,
        OutputExprToResultSubPCGMapping
      > result = evaluate_substitution_output(pcg, sub, match);

      SubParallelComputationGraph result_graph = result.first;
      bidict<parallel_layer_guid_t, OutputGraphExprNode> result_node_map = result.second.node_mapping;
      bidict<input_parallel_tensor_guid_t, OutputGraphExprInput> result_input_map = result.second.input_mapping;

      LinearAttrs correct_result_fused_mm_relu_attrs = LinearAttrs{
        12,
        /*use_bias=*/false,
        DataType::FLOAT,
        Activation::RELU,
        /*regularizer=*/std::nullopt,
      };

      ParallelTensorAttrs correct_result_i_activation_attrs = get_parallel_tensor_attrs(pcg, mm_match_layer_input_activations);
      ParallelTensorAttrs correct_result_i_weights_attrs = get_parallel_tensor_attrs(pcg, mm_match_layer_input_weights);
      ParallelTensorAttrs correct_result_fused_mm_relu_output_attrs = get_parallel_tensor_attrs(pcg, open_parallel_tensor_guid_from_closed(get_only(get_layer_outputs(pcg, relu_match_layer))));

      parallel_layer_guid_t result_fused_mm_relu_node = result_node_map.at_r(fused_mm_relu_node);
      parallel_tensor_guid_t result_fused_mm_relu_output = get_only(get_layer_outputs(result_graph, result_fused_mm_relu_node));
      input_parallel_tensor_guid_t result_i_activation = result_input_map.at_r(output_i_activation);
      input_parallel_tensor_guid_t result_i_weights = result_input_map.at_r(output_i_weights);

      SubParallelComputationGraphData correct_graph_data = SubParallelComputationGraphData{
        std::unordered_map<parallel_layer_guid_t, ParallelLayerAttrs>{
          {
            result_fused_mm_relu_node,
            ParallelLayerAttrs{
              PCGOperatorAttrs{correct_result_fused_mm_relu_attrs},
              /*name=*/std::nullopt,
            },
          }
        },
        std::unordered_set<SubParallelComputationGraphEdge>{
          SubParallelComputationGraphEdge{
            OpenDataflowEdge{
              DataflowInputEdge{
                result_i_activation.raw_dataflow_graph_input,
                DataflowInput{
                  result_fused_mm_relu_node.raw_graph_node,
                  0,
                },
              },
            },
          },
          SubParallelComputationGraphEdge{
            OpenDataflowEdge{
              DataflowInputEdge{
                result_i_weights.raw_dataflow_graph_input,
                DataflowInput{
                  result_fused_mm_relu_node.raw_graph_node,
                  1,
                },
              },
            },
          },
        },
        std::unordered_set<input_parallel_tensor_guid_t>{
          result_i_activation,
          result_i_weights,
        },
        std::unordered_map<open_parallel_tensor_guid_t, ParallelTensorAttrs>{
          {
            open_parallel_tensor_guid_from_input(result_i_activation),
            correct_result_i_activation_attrs,
          },
          {
            open_parallel_tensor_guid_from_input(result_i_weights),
            correct_result_i_weights_attrs,
          },
          {
            open_parallel_tensor_guid_from_closed(result_fused_mm_relu_output),
            correct_result_fused_mm_relu_output_attrs,                     
          }
        }
      };

      SubParallelComputationGraphData result_graph_data = get_sub_pcg_data(result_graph);

      CHECK(result_graph_data == correct_graph_data);
    }

    SUBCASE("apply_substitution") {
      SubParallelComputationGraph result = apply_substitution(pcg, sub, match);

      SubParallelComputationGraph correct = [&] {
        ParallelComputationGraphBuilder b;
        parallel_tensor_guid_t t = b.create_input_tensor(
          ParallelTensorShape{
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
          }
        );
        t = b.dense(t, 
                    /*outDim=*/16,
                    /*activation=*/std::nullopt);
        t = b.gelu(t);
        t = b.dense(t,
                    /*outDim=*/12,
                    /*activation=*/Activation::RELU,
                    /*use_bias=*/false,
                    /*data_type=*/DataType::FLOAT,
                    /*kernel_initializer=*/std::nullopt,
                    /*bias_initializer=*/std::nullopt,
                    /*name=*/std::nullopt);
        t = b.dense(t, 
                    /*outDim=*/8,
                    /*activation=*/Activation::RELU);
      
        return sub_pcg_from_full_pcg(b.pcg);
      }();

      CHECK(get_parallel_layers(result).size() == get_parallel_layers(correct).size());
      CHECK(get_edges(result.raw_graph).size() == get_edges(correct.raw_graph).size());
      // CHECK(are_isomorphic(result, correct));
    }
  }
}
