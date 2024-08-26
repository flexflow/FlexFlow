#include "substitutions/substitution.h"
#include "op-attrs/get_output_shapes.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.h"
#include "substitutions/open_parallel_tensor_guid_t.h"
#include "substitutions/output_expr_to_result_sub_pcg_mapping.dtg.h"
#include "substitutions/output_expr_to_result_sub_pcg_mapping.h"
#include "substitutions/output_graph/output_operator_attrs_assignment.h"
#include "substitutions/pcg_pattern_match.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "substitutions/sub_parallel_computation_graph_edge.h"
#include "utils/bidict/algorithms/right_entries.h"
#include "utils/containers/map_values.h"
#include "utils/containers/merge_maps.h"
#include "utils/containers/restrict_keys.h"
#include "utils/containers/set_minus.h"
#include "utils/containers/transform.h"
#include "utils/containers/values.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/graph/instances/unordered_set_labelled_open_dataflow_graph.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/permute_input_ids.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/permute_node_ids.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/rewrite_labels.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/rewrite_node_labels.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/rewrite_value_labels.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/node/algorithms/generate_new_node_id_permutation.h"
#include "utils/graph/open_dataflow_graph/algorithms/generate_new_input_id_permutation.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_open_dataflow_value_uses.h"
#include "utils/overload.h"

namespace FlexFlow {

bool is_valid_substitution(Substitution const &) {
  NOT_IMPLEMENTED();
}

LabelledOpenDataflowGraphView<ParallelLayerAttrs, ParallelTensorShape>
    perform_shape_inference(
        LabelledOpenDataflowGraphView<ParallelLayerAttrs, std::monostate> const
            &g,
        std::unordered_map<DataflowGraphInput, ParallelTensorShape> const
            &input_shapes) {

  std::unordered_map<OpenDataflowValue, ParallelTensorShape> inferred =
      map_keys(input_shapes, [](DataflowGraphInput const &i) {
        return OpenDataflowValue{i};
      });

  for (Node const &n : get_topological_ordering(g)) {
    std::vector<ParallelTensorShape> input_shapes =
        transform(get_inputs(g, n),
                  [&](OpenDataflowValue const &v) { return inferred.at(v); });
    std::vector<ParallelTensorShape> output_shapes =
        get_output_shapes(g.at(n).op_attrs, input_shapes);
    std::vector<DataflowOutput> outputs = get_outputs(g, n);

    for (auto const &[output, shape] : zip(outputs, output_shapes)) {
      inferred.insert({OpenDataflowValue{output}, shape});
    }
  }

  return rewrite_value_labels(
      g, [&](OpenDataflowValue const &v, std::monostate const &) {
        return inferred.at(v);
      });
}

std::pair<SubParallelComputationGraph, OutputExprToResultSubPCGMapping>
    evaluate_substitution_output(SubParallelComputationGraph const &spcg,
                                 Substitution const &sub,
                                 PCGPatternMatch const &match) {
  std::unordered_map<PatternNode, PCGOperatorAttrs> node_match =
      map_values(match.node_assignment.as_unordered_map(),
                 [&](parallel_layer_guid_t const &n) {
                   return get_operator_attrs(spcg, n);
                 });

  bidict<NewNode, Node> new_node_id_permutation =
      generate_new_node_id_permutation(sub.output_graph_expr.raw_graph);
  bidict<NewDataflowGraphInput, DataflowGraphInput> new_input_id_permutation =
      generate_new_input_id_permutation(sub.output_graph_expr.raw_graph);
  LabelledOpenDataflowGraphView<OutputOperatorAttrsAssignment, std::monostate>
      permuted =
          permute_input_ids(permute_node_ids(sub.output_graph_expr.raw_graph,
                                             new_node_id_permutation),
                            new_input_id_permutation);

  LabelledOpenDataflowGraphView<ParallelLayerAttrs, std::monostate>
      without_shapes = rewrite_node_labels(
          permuted,
          [&](Node const &n, OutputOperatorAttrsAssignment const &attrs) {
            return ParallelLayerAttrs{
                materialize_output_operator_from_attrs_assignment(attrs,
                                                                  node_match),
                std::nullopt,
            };
          });

  bidict<input_parallel_tensor_guid_t, OutputGraphExprInput> result_input_map =
      map_keys(map_values(new_input_id_permutation,
                          [](DataflowGraphInput const &i) {
                            return OutputGraphExprInput{i};
                          }),
               [](NewDataflowGraphInput const &i) {
                 return input_parallel_tensor_guid_t{i.raw_input};
               });

  bidict<parallel_layer_guid_t, OutputGraphExprNode> result_node_map = map_keys(
      map_values(new_node_id_permutation,
                 [](Node const &n) { return OutputGraphExprNode{n}; }),
      [](NewNode const &n) { return parallel_layer_guid_t{n.raw_node}; });

  std::unordered_map<DataflowGraphInput, ParallelTensorShape> input_shapes =
      map_values(map_keys(match.input_assignment,
                          [&](PatternInput const &i) {
                            return result_input_map
                                .at_r(sub.inputs_mapping.at_l(i))
                                .raw_dataflow_graph_input;
                          }),
                 [&](open_parallel_tensor_guid_t const &v) {
                   return spcg.raw_graph.at(v.raw_open_dataflow_value).shape;
                 });
  LabelledOpenDataflowGraphView<ParallelLayerAttrs, ParallelTensorShape>
      with_shapes = perform_shape_inference(without_shapes, input_shapes);
  LabelledOpenDataflowGraphView<ParallelLayerAttrs, ParallelTensorAttrs>
      with_attrs = rewrite_value_labels(
          with_shapes,
          [](OpenDataflowValue const &, ParallelTensorShape const &s) {
            return ParallelTensorAttrs{
                s,
                std::nullopt,
                std::nullopt,
                CreateGrad::YES,
            };
          });

  return std::make_pair(SubParallelComputationGraph{with_attrs},
                        OutputExprToResultSubPCGMapping{
                            result_node_map,
                            result_input_map,
                        });
}

SubParallelComputationGraph
    apply_substitution(SubParallelComputationGraph const &spcg,
                       Substitution const &sub,
                       PCGPatternMatch const &match) {
  auto substitution_output_result =
      evaluate_substitution_output(spcg, sub, match);
  SubParallelComputationGraph substitution_output_graph =
      substitution_output_result.first;
  OutputExprToResultSubPCGMapping output_expr_to_result_sub_pcg_mapping =
      substitution_output_result.second;

  SubParallelComputationGraphData output_graph_data =
      get_sub_pcg_data(substitution_output_graph);
  SubParallelComputationGraphData pre_data = get_sub_pcg_data(spcg);

  std::unordered_set<parallel_layer_guid_t> pre_nodes =
      keys(pre_data.node_data);
  std::unordered_set<parallel_layer_guid_t> matched_nodes =
      unordered_set_of(values(match.node_assignment));
  std::unordered_set<parallel_layer_guid_t> post_nodes_from_original_graph =
      set_minus(pre_nodes, matched_nodes);

  std::unordered_map<parallel_layer_guid_t, ParallelLayerAttrs> post_node_data =
      [&] {
        std::unordered_map<parallel_layer_guid_t, ParallelLayerAttrs>
            post_node_data_from_orig = restrict_keys(
                pre_data.node_data, post_nodes_from_original_graph);
        std::unordered_map<parallel_layer_guid_t, ParallelLayerAttrs>
            post_node_data_from_sub = output_graph_data.node_data;

        return merge_maps(post_node_data_from_orig, post_node_data_from_sub);
      }();

  std::unordered_set<SubParallelComputationGraphEdge> post_edges = [&] {
    std::unordered_set<SubParallelComputationGraphEdge> post_edges_from_orig =
        filter(pre_data.edges, [&](SubParallelComputationGraphEdge const &e) {
          if (e.raw_edge.has<DataflowInputEdge>()) {
            return true;
          } else {
            DataflowEdge dfe = e.raw_edge.get<DataflowEdge>();
            parallel_layer_guid_t src = parallel_layer_guid_t{dfe.src.node};
            parallel_layer_guid_t dst = parallel_layer_guid_t{dfe.dst.node};
            return !(contains(matched_nodes, src) ||
                     contains(matched_nodes, dst));
          }
        });

    std::unordered_set<SubParallelComputationGraphEdge> post_edges_from_sub =
        filter(output_graph_data.edges,
               [&](SubParallelComputationGraphEdge const &e) {
                 return !e.raw_edge.has<DataflowInputEdge>();
               });

    bidict<PatternNodeOutput, parallel_tensor_guid_t>
        output_orig_pattern_mapping = get_output_mapping_for_pcg_pattern_match(
            match, sub.pcg_pattern, spcg);
    bidict<parallel_tensor_guid_t, OutputGraphExprNodeOutput>
        output_post_outexpr_mapping = get_output_graph_expr_output_mapping(
            output_expr_to_result_sub_pcg_mapping,
            sub.output_graph_expr,
            substitution_output_graph);

    std::unordered_set<SubParallelComputationGraphEdge> incoming_to_sub_edges;
    for (auto const &[pattern_input, base_graph_tensor] :
         match.input_assignment) {
      OutputGraphExprInput output_expr_input =
          sub.inputs_mapping.at_l(pattern_input);
      input_parallel_tensor_guid_t output_graph_input =
          output_expr_to_result_sub_pcg_mapping.input_mapping.at_r(
              output_expr_input);
      std::unordered_set<parallel_tensor_use_t> uses = get_parallel_tensor_uses(
          substitution_output_graph,
          open_parallel_tensor_guid_from_input(output_graph_input));
      for (parallel_tensor_use_t const &use : uses) {
        SubParallelComputationGraphEdge new_edge =
            subpcg_edge_from_tensor_and_use(base_graph_tensor, use);
        incoming_to_sub_edges.insert(new_edge);
      }
    }

    std::unordered_set<SubParallelComputationGraphEdge> outgoing_from_sub_edges;
    for (ParallelComputationGraphEdge const &outgoing_edge :
         get_outgoing_edges(spcg, matched_nodes, IncludeInternalEdges::NO)) {
      parallel_tensor_guid_t original_tensor =
          get_parallel_tensor(outgoing_edge);
      PatternNodeOutput pattern_tensor =
          output_orig_pattern_mapping.at_r(original_tensor);
      OutputGraphExprNodeOutput output_graph_tensor =
          sub.outputs_mapping.at_l(pattern_tensor);
      parallel_tensor_guid_t new_tensor =
          output_post_outexpr_mapping.at_r(output_graph_tensor);

      SubParallelComputationGraphEdge new_edge =
          subpcg_edge_from_tensor_and_dst(
              new_tensor,
              get_dst_layer(outgoing_edge),
              get_dst_layer_input_idx(outgoing_edge));
      outgoing_from_sub_edges.insert(new_edge);
    }

    return set_union(std::vector{
        post_edges_from_orig,
        post_edges_from_sub,
        incoming_to_sub_edges,
        outgoing_from_sub_edges,
    });
  }();

  std::unordered_set<input_parallel_tensor_guid_t> post_inputs =
      pre_data.inputs;

  std::unordered_map<open_parallel_tensor_guid_t, ParallelTensorAttrs>
      post_value_data = [&] {
        std::unordered_map<open_parallel_tensor_guid_t, ParallelTensorAttrs>
            post_value_data_from_orig = filter_keys(
                pre_data.value_data, [&](open_parallel_tensor_guid_t const &t) {
                  return visit_open_parallel_tensor_guid(
                      t,
                      overload{
                          [&](parallel_tensor_guid_t const &t) {
                            return contains(post_nodes_from_original_graph,
                                            get_source_layer(t));
                          },
                          [](input_parallel_tensor_guid_t const &) {
                            return true;
                          },
                      });
                });

        std::unordered_map<open_parallel_tensor_guid_t, ParallelTensorAttrs>
            post_value_data_from_sub = output_graph_data.value_data;
        return merge_maps(post_value_data_from_orig, post_value_data_from_sub);
      }();

  SubParallelComputationGraphData post_data = SubParallelComputationGraphData{
      post_node_data,
      post_edges,
      post_inputs,
      post_value_data,
  };

  return sub_pcg_from_graph_data(post_data);
}

} // namespace FlexFlow
