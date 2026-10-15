#include "compiler/machine_mapping/apply_substitution_and_update_machine_mapping.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.h"
#include "substitutions/apply_substitution/apply_substitution.h"
#include "substitutions/apply_substitution/evaluate_substitution_output.h"
#include "substitutions/apply_substitution/output_expr_to_result_sub_pcg_mapping.h"
#include "substitutions/open_parallel_tensor_guid_t.h"
#include "substitutions/pcg_pattern_match.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "substitutions/sub_parallel_computation_graph_data.dtg.h"
#include "substitutions/sub_parallel_computation_graph_edge.h"
#include "utils/containers/binary_merge_disjoint_maps.h"
#include "utils/containers/filter.h"
#include "utils/containers/is_subseteq_of.h"
#include "utils/containers/keys.h"
#include "utils/containers/set_minus.h"
#include "utils/containers/values.h"
#include "utils/containers/vector_of.h"
#include "utils/random_utils.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

SearchResult apply_substitution_and_update_machine_mapping(
    std::mt19937 &gen,
    SearchResult const &mapped_pcg,
    Substitution const &sub,
    PCGPatternMatch const &match) {
  SubParallelComputationGraph spcg = sub_pcg_from_full_pcg(mapped_pcg.pcg);

  std::pair<SubParallelComputationGraph, OutputExprToResultSubPCGMapping>
      substitution_output_result =
          evaluate_substitution_output(spcg, sub, match);

  SubParallelComputationGraph post_substitution_graph =
      apply_substitution_from_output_result(
          substitution_output_result, spcg, sub, match);

  std::map<parallel_layer_guid_t, ParallelLayerAttrs> post_node_data =
      get_sub_pcg_data(post_substitution_graph).node_data;

  std::set<parallel_layer_guid_t> substitution_output_parallel_layers =
      spcg_get_parallel_layers(substitution_output_result.first);

  std::map<parallel_layer_guid_t, MachineView> machine_views =
      mapped_pcg.machine_mapping.machine_views;

  std::set<parallel_layer_guid_t> matched_nodes =
      set_of(values(match.node_assignment));

  std::vector<MachineView> substituted_machine_views = vector_of(
      transform(matched_nodes, [&](parallel_layer_guid_t const &node) {
        return machine_views.at(node);
      }));

  for (parallel_layer_guid_t layer : substitution_output_parallel_layers) {
    machine_views.insert_or_assign(
        layer, select_random(gen, substituted_machine_views));
  }

  ASSERT(is_subseteq_of(keys(post_node_data), keys(machine_views)));

  std::map<parallel_layer_guid_t, MachineView> post_node_machine_views =
      filter(machine_views,
             [&](std::pair<parallel_layer_guid_t, MachineView> const &p) {
               return post_node_data.count(p.first);
             });

  ASSERT(keys(post_node_data) == keys(post_node_machine_views));

  return SearchResult{
      pcg_from_sub_pcg_by_dropping_inputs(post_substitution_graph),
      MachineMapping{post_node_machine_views}};
}

} // namespace FlexFlow
