#include "compiler/machine_mapping/get_optimal_machine_mapping.h"
#include "compiler/machine_mapping/get_tensor_set_movement_across_split.h"
#include "compiler/machine_mapping/get_allowed_machine_views_list.h"
#include "compiler/machine_mapping/get_machine_resource_splits.h"
#include "compiler/machine_mapping/machine_mapping_result.h"
#include "compiler/machine_mapping/partial_machine_mapping.dtg.h"
#include "compiler/machine_mapping/partial_machine_mapping.h"
#include "compiler/machine_mapping/transitive_reduced_pcg.h"
#include "compiler/series_parallel/pcg_binary_sp_decomposition.dtg.h"
#include "compiler/series_parallel/pcg_binary_sp_decomposition.h"
#include "pcg/machine_specification.dtg.h"
#include "pcg/machine_specification.h"
#include "pcg/machine_view.dtg.h"
#include "pcg/machine_view.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "utils/containers/contains.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/get_all_assignments.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/exception.h"
#include "utils/overload.h"
#include "compiler/series_parallel/pcg_binary_parallel_split.dtg.h"
#include "compiler/series_parallel/pcg_binary_parallel_split.h"
#include "compiler/series_parallel/pcg_binary_series_split.h"
#include "compiler/machine_mapping/machine_mapping_context.h"
#include "utils/containers/flatmap.h"
#include "compiler/machine_mapping/estimate_layer_cost.h"


namespace FlexFlow {

MachineMappingResult get_optimal_machine_mapping(
    ParallelComputationGraph const &pcg,
    std::function<std::unordered_set<MachineView>(
        ParallelLayerAttrs const &, MachineSpecification const &)> const
        &allowed_machine_views,
    CostEstimator const &cost_estimator,
    MachineSpecification const &resources,
    MachineMappingCache &result_cache) {

  MachineMappingContext context = make_machine_mapping_context(
      pcg, 
      cost_estimator, 
      allowed_machine_views);

  MachineMappingResult result =
      get_optimal_machine_mapping_internal(result_cache, context, resources);

  return result;
}

MachineMappingResult get_optimal_machine_mapping_internal(
    MachineMappingCache &result_cache,                                                          
    MachineMappingContext const &context, 
    MachineSpecification const &resources) {

  PCGBinarySPDecomposition sp_decomposition_tree = ({
    std::optional<PCGBinarySPDecomposition> returned = get_pcg_balanced_binary_sp_decomposition(context.transitive_reduced_pcg.full_pcg);
    if (!returned.has_value()) {
      throw mk_runtime_error("Failed to get serial parallel decomposition");
    }
    returned.value();
  });

  std::unordered_set<parallel_layer_guid_t> all_layers = get_parallel_layers(context.transitive_reduced_pcg.full_pcg);

  return get_optimal_machine_mapping_internal(result_cache, 
                                              context, 
                                              sp_decomposition_tree, 
                                              resources,
                                              get_unconstrained_solution_for_layers(all_layers));
}

MachineMappingResult get_optimal_machine_mapping_internal(
    MachineMappingCache &result_cache,
    MachineMappingContext const &context,
    PCGBinarySPDecomposition const &sp_decomposition_tree,
    MachineSpecification const &resources,
    PartialMachineMapping const &partial_solution) {

  MachineMappingState state = MachineMappingState{
    sp_decomposition_tree, resources, partial_solution,
  };

  {
    std::optional<MachineMappingResult> cached_result =
        result_cache.load(state);
    if (cached_result) {
      return cached_result.value();
    }
  }

  MachineMappingResult result = visit<MachineMappingResult>(
    sp_decomposition_tree,
    [&](auto const &decomp_tree_node) {
      return get_optimal_machine_mapping_internal(result_cache, context, decomp_tree_node, resources, partial_solution);
    });

  result_cache.save(state, result);
  return result;
}

MachineMappingResult get_optimal_machine_mapping_internal(
    MachineMappingCache &result_cache,
    MachineMappingContext const &context,
    PCGBinarySeriesSplit const &series_split,
    MachineSpecification const &resource,
    PartialMachineMapping const &partial_solution) {

  MachineMappingResult optimal_result = get_infinity_machine_mapping_result();

  auto is_subgraph_input = [&](std::unordered_set<Node> const &subgraph_nodes,
                               parallel_tensor_guid_t const &input_tensor) {
    return !contains(subgraph_nodes, input_tensor.raw_graph_output.node);
  };

  PCGBinarySPDecomposition pre_sub_tree = get_left_child(series_split);
  PCGBinarySPDecomposition post_sub_tree = get_right_child(series_split);

  PCGSplitBoundaryLayers boundary_layers = 
    pcg_get_transitive_reduced_boundary_layers_for_split(context.transitive_reduced_pcg, 
                                                         series_split);

  auto get_boundary_machine_view_assignments = [&](std::unordered_set<parallel_layer_guid_t> const &layers) 
    -> std::unordered_set<std::unordered_map<parallel_layer_guid_t, MachineView>>
  {
    std::unordered_map<parallel_layer_guid_t, std::unordered_set<MachineView>>
      allowed = generate_map(layers,
                             [&](parallel_layer_guid_t const &l) { 
                               return get_allowed_machine_views_for_layer(context, l);
                             });
    return get_all_assignments(allowed);
  };

  for (std::unordered_map<parallel_layer_guid_t, MachineView> const &assigned_pre_machine_views
        : get_boundary_machine_view_assignments(boundary_layers.pre_split_boundary)) {

    PartialMachineMapping pre_candidate = 
      with_additional_layer_machine_views(
        get_sub_solution(partial_solution, pre_sub_tree),
        assigned_pre_machine_views);

    MachineMappingResult pre_result =
      get_optimal_machine_mapping_internal(result_cache, 
                                           context, 
                                           pre_sub_tree,
                                           resource, 
                                           pre_candidate);


    for (std::unordered_map<parallel_layer_guid_t, MachineView> const &assigned_post_machine_views
          : get_boundary_machine_view_assignments(boundary_layers.post_split_boundary)) {

      PartialMachineMapping post_candidate = 
        with_additional_layer_machine_views(
          get_sub_solution(partial_solution, post_sub_tree),
          assigned_post_machine_views);

      MachineMappingResult post_result =
        get_optimal_machine_mapping_internal(result_cache, 
                                             context, 
                                             post_sub_tree,
                                             resource, 
                                             post_candidate);

      TensorSetMovement comm_across_split = get_tensor_set_movement_across_split(
        /*transitive_reduced_pcg=*/context.transitive_reduced_pcg,
        /*split=*/series_split,
        /*pre_mapping=*/pre_candidate,
        /*post_mapping=*/post_candidate);

      float cost_across_split = context.cost_estimator.estimate_cost(comm_across_split);

      minimize_runtime(
        optimal_result,
        sequential_combine(pre_result, cost_across_split, post_result));
    }
  }

  return optimal_result;
}



MachineMappingResult get_optimal_machine_mapping_internal(
    MachineMappingCache &result_cache,                                                          
    MachineMappingContext const &context,
    PCGBinaryParallelSplit const &parallel,
    MachineSpecification const &resources,
    PartialMachineMapping const &partial_solution) {

  PCGBinarySPDecomposition left_subtree = get_left_child(parallel);
  PartialMachineMapping left_sub_solution = get_sub_solution(partial_solution,
                                                             left_subtree);

  PCGBinarySPDecomposition right_subtree = get_right_child(parallel);
  PartialMachineMapping right_sub_solution = get_sub_solution(partial_solution, 
                                                              right_subtree);

  MachineMappingResult optimal_result = [&] {
    PCGBinarySeriesSplit series = require_series(make_pcg_series_split(
      get_left_child(parallel),
      get_right_child(parallel)));
    return get_optimal_machine_mapping_internal(result_cache,
                                                context,
                                                series,
                                                resources,
                                                partial_solution);
  }();

  for (auto const &resource_split : get_machine_resource_splits(resources)) {
    MachineMappingResult left_result = 
      get_optimal_machine_mapping_internal(result_cache,
                                           context,
                                           left_subtree,
                                           resource_split.first,
                                           left_sub_solution);
    MachineMappingResult right_result = 
      get_optimal_machine_mapping_internal(result_cache,
                                           context,
                                           right_subtree,
                                           resource_split.second,
                                           right_sub_solution);

    minimize_runtime(
        optimal_result,
        parallel_combine(left_result, right_result));
  }

  return optimal_result;
}

MachineMappingResult get_optimal_machine_mapping_internal(
    MachineMappingCache &result_cache,
    MachineMappingContext const &context,
    parallel_layer_guid_t const &layer,
    MachineSpecification const &resource,
    PartialMachineMapping const &partial_solution) {

  assert (get_all_layers(partial_solution, IncludeUnconstrained{true}) == std::unordered_set{layer});

  float cost = estimate_layer_cost(context.transitive_reduced_pcg.full_pcg, 
                                   context.cost_estimator,
                                   layer, 
                                   get_machine_view_for_layer(partial_solution, layer).value());

  return MachineMappingResult{
    /*runtime=*/cost,
    /*machine_mapping=*/require_complete_mapping(partial_solution),
  };
}

} // namespace FlexFlow
