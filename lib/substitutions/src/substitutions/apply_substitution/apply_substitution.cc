#include "substitutions/apply_substitution/apply_substitution.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.h"
#include "substitutions/apply_substitution/evaluate_substitution_output.h"
#include "substitutions/apply_substitution/output_expr_to_result_sub_pcg_mapping.h"
#include "substitutions/open_parallel_tensor_guid_t.h"
#include "substitutions/pcg_pattern_match.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "substitutions/sub_parallel_computation_graph_data.dtg.h"
#include "substitutions/sub_parallel_computation_graph_data.h"
#include "substitutions/sub_parallel_computation_graph_edge.h"
#include "utils/containers/binary_merge_disjoint_maps.h"
#include "utils/containers/keys.h"
#include "utils/containers/restrict_keys.h"
#include "utils/containers/set_minus.h"
#include "utils/containers/values.h"

namespace FlexFlow {

SubParallelComputationGraph
    apply_substitution(SubParallelComputationGraph const &spcg,
                       Substitution const &sub,
                       PCGPatternMatch const &match) {
  assert_pcg_pattern_match_is_valid_for_pattern_and_subpcg(
      match, sub.pcg_pattern, spcg);
  std::pair<SubParallelComputationGraph, OutputExprToResultSubPCGMapping>
      substitution_output_result =
          evaluate_substitution_output(spcg, sub, match);
  return apply_substitution_from_output_result(
      substitution_output_result, spcg, sub, match);
}

SubParallelComputationGraph apply_substitution_from_output_result(
    std::pair<SubParallelComputationGraph,
              OutputExprToResultSubPCGMapping> const
        &substitution_output_result,
    SubParallelComputationGraph const &spcg,
    Substitution const &sub,
    PCGPatternMatch const &match) {
  SubParallelComputationGraph substitution_output_graph =
      substitution_output_result.first;
  OutputExprToResultSubPCGMapping output_expr_to_result_sub_pcg_mapping =
      substitution_output_result.second;

  SubParallelComputationGraphData output_graph_data =
      get_sub_pcg_data(substitution_output_graph);
  require_sub_parallel_computation_graph_data_is_valid(output_graph_data);

  SubParallelComputationGraphData pre_data = get_sub_pcg_data(spcg);
  require_sub_parallel_computation_graph_data_is_valid(pre_data);

  std::set<parallel_layer_guid_t> pre_nodes = keys(pre_data.node_data);
  std::set<parallel_layer_guid_t> matched_nodes =
      match.node_assignment.right_values();
  std::set<parallel_layer_guid_t> post_nodes_from_original_graph =
      set_minus(pre_nodes, matched_nodes);

  std::map<parallel_layer_guid_t, ParallelLayerAttrs> post_node_data = [&] {
    std::map<parallel_layer_guid_t, ParallelLayerAttrs>
        post_node_data_from_orig =
            restrict_keys(pre_data.node_data, post_nodes_from_original_graph);
    std::map<parallel_layer_guid_t, ParallelLayerAttrs>
        post_node_data_from_sub = output_graph_data.node_data;

    return binary_merge_disjoint_maps(post_node_data_from_orig,
                                      post_node_data_from_sub);
  }();

  std::set<input_parallel_tensor_guid_t> post_inputs = pre_data.inputs;

  std::set<SubParallelComputationGraphEdge> post_edges = [&] {
    std::set<SubParallelComputationGraphEdge> post_edges_from_orig =
        filter(pre_data.edges, [&](SubParallelComputationGraphEdge const &e) {
          if (e.raw_edge.is_input_edge()) {
            return true;
          } else {
            KwargDataflowEdge<TensorSlotName> dfe =
                e.raw_edge.require_internal_edge();
            parallel_layer_guid_t src = parallel_layer_guid_t{dfe.src.node};
            parallel_layer_guid_t dst = parallel_layer_guid_t{dfe.dst.node};
            return !(contains(matched_nodes, src) ||
                     contains(matched_nodes, dst));
          }
        });

    std::set<SubParallelComputationGraphEdge> post_edges_from_sub = filter(
        output_graph_data.edges, [&](SubParallelComputationGraphEdge const &e) {
          return e.raw_edge.is_internal_edge();
        });

    bidict<PatternNodeOutput, parallel_tensor_guid_t>
        output_orig_pattern_mapping = get_output_mapping_for_pcg_pattern_match(
            match, sub.pcg_pattern, spcg);
    bidict<parallel_tensor_guid_t, OutputGraphExprNodeOutput>
        output_post_outexpr_mapping = get_output_graph_expr_output_mapping(
            output_expr_to_result_sub_pcg_mapping,
            sub.output_graph_expr,
            substitution_output_graph);

    std::set<SubParallelComputationGraphEdge> incoming_to_sub_edges;
    for (auto const &[pattern_input, base_graph_tensor] :
         match.input_assignment) {
      OutputGraphExprInput output_expr_input =
          sub.inputs_mapping.at_l(pattern_input);
      input_parallel_tensor_guid_t output_graph_input =
          output_expr_to_result_sub_pcg_mapping.input_mapping.at_r(
              output_expr_input);
      std::set<parallel_tensor_use_t> uses = get_open_parallel_tensor_uses(
          substitution_output_graph,
          open_parallel_tensor_guid_from_input(output_graph_input));
      for (parallel_tensor_use_t const &use : uses) {
        SubParallelComputationGraphEdge new_edge =
            subpcg_edge_from_tensor_and_use(base_graph_tensor, use);
        incoming_to_sub_edges.insert(new_edge);
      }
    }

    std::set<SubParallelComputationGraphEdge> outgoing_from_sub_edges;
    for (ParallelComputationGraphEdge const &outgoing_edge :
         get_subgraph_outgoing_edges(spcg, matched_nodes)) {
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
              get_dst_layer_input_slot_name(outgoing_edge));
      outgoing_from_sub_edges.insert(new_edge);
    }

    return set_union(std::vector{
        post_edges_from_orig,
        post_edges_from_sub,
        incoming_to_sub_edges,
        outgoing_from_sub_edges,
    });
  }();

  std::map<open_parallel_tensor_guid_t, ParallelTensorAttrs> post_value_data =
      [&] {
        std::map<open_parallel_tensor_guid_t, ParallelTensorAttrs>
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

        std::map<open_parallel_tensor_guid_t, ParallelTensorAttrs>
            post_value_data_from_sub = output_graph_data.value_data;
        return binary_merge_disjoint_maps(post_value_data_from_orig,
                                          post_value_data_from_sub);
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
