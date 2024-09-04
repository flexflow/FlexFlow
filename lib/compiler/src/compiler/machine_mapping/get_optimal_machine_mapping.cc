#include "compiler/machine_mapping/get_optimal_machine_mapping.h"
#include "compiler/cost_estimator.h"
#include "compiler/machine_mapping/machine_mapping_result.h"
#include "compiler/machine_mapping/split_sp_decomposition.h"
#include "compiler/machine_mapping/allowed_machine_mappings.h"
#include "pcg/machine_specification.dtg.h"
#include "pcg/machine_specification.h"
#include "pcg/machine_view.dtg.h"
#include "pcg/machine_view.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "utils/containers.h"
#include "utils/containers/as_vector.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/get_only.h"
#include "utils/containers/keys.h"
#include "utils/containers/restrict_keys.h"
#include "utils/containers/values.h"
#include "utils/exception.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/graph_split.dtg.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/open_dataflow_graph/algorithms.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_subgraph.h"
#include "utils/graph/serial_parallel/get_serial_parallel_decomposition.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.h"
#include "utils/graph/serial_parallel/serial_parallel_splits.h"
#include "utils/overload.h"

namespace FlexFlow {

std::vector<std::pair<MachineSpecification, MachineSpecification>>
    get_resource_split(MachineSpecification const &resource) {
  std::vector<std::pair<MachineSpecification, MachineSpecification>> result;
  for (int i = 1; i < resource.num_nodes; ++i) {
    MachineSpecification sub_resource1 = resource, sub_resource2 = resource;
    sub_resource1.num_nodes = i;
    sub_resource2.num_nodes = resource.num_nodes - i;
    result.push_back(std::make_pair(sub_resource1, sub_resource2));
  }
  return result;
}

GraphSplit
    get_graph_split(SerialParallelDecomposition const &pre_decomposition,
                    SerialParallelDecomposition const &post_decomposition) {
  return GraphSplit{get_nodes(pre_decomposition),
                    get_nodes(post_decomposition)};
}

float base_case_estimate_cost(
    SubParallelComputationGraph const &g,
    CostEstimator const &estimator,
    std::unordered_map<OpenDataflowValue, MachineView> const &machine_views) {
  // In the base case, all the operators are executed sequentially.
  float cost = 0.1;
  // TODO(@wmdi)
  return cost;
}

MachineMappingResult get_optimal_machine_mapping(
    ParallelComputationGraph const &pcg,
    std::function<std::unordered_set<MachineView>(
        ParallelLayerAttrs const &, MachineSpecification const &)> const
        &allowed_machine_views,
    CostEstimator const &cost_estimator,
    MachineSpecification const &resources,
    MachineMappingCache &cached_subgraph_results) {

  MachineMappingContext context(
      pcg, cost_estimator, allowed_machine_views, cached_subgraph_results);
  MachineMappingResult result =
      get_optimal_machine_mapping_internal(context, resources);
  cached_subgraph_results = context.cached_subgraph_results;
  return result;
}

MachineMappingResult get_optimal_machine_mapping_internal(
    MachineMappingContext &context, MachineSpecification const &resources) {
  std::optional<SerialParallelDecomposition> decompn_optional =
      get_serial_parallel_decomposition(context.pcg.raw_graph);

  if (!decompn_optional) {
    throw mk_runtime_error("Failed to get serial parallel decomposition");
  }

  SerialParallelDecomposition decompn = decompn_optional.value();

  return get_optimal_machine_mapping_internal(context, decompn, resources, {});
}

MachineMappingResult get_optimal_machine_mapping_internal(
    MachineMappingContext &context,
    SerialParallelDecomposition const &decompn,
    MachineSpecification const &resource,
    std::unordered_map<OpenDataflowValue, MachineView> const
        &fixed_machine_views) {

  MachineMappingState state(decompn, resource, fixed_machine_views);
  std::optional<MachineMappingResult> cached_result =
      context.cached_subgraph_results.load(state);
  if (cached_result) {
    return cached_result.value();
  }

  MachineMappingResult result = decompn.visit<MachineMappingResult>(
      overload{[&](SerialSplit const &serial) {
                 return get_optimal_machine_mapping_internal(
                     context, serial, resource, fixed_machine_views);
               },
               [&](ParallelSplit const &parallel) {
                 return get_optimal_machine_mapping_internal(
                     context, parallel, resource, fixed_machine_views);
               },
               [&](Node const &node) {
                 return get_optimal_machine_mapping_internal(
                     context, node, resource, fixed_machine_views);
               }});

  context.cached_subgraph_results.save(state, result);
  return result;
}

MachineMappingResult get_optimal_machine_mapping_internal(
    MachineMappingContext &context,
    SerialSplit const &serial,
    MachineSpecification const &resource,
    std::unordered_map<OpenDataflowValue, MachineView> const
        &fixed_machine_views) {
  MachineMappingResult optimal_result = get_infinity_machine_mapping_result();

  auto [decompn1, decompn2] = split_sp_decomposition(serial);

  GraphSplit graph_split = get_graph_split(decompn1, decompn2);

  OpenDataflowSubgraphResult subgraph_res1 = get_subgraph(
      sub_pcg_from_full_pcg(context.pcg).raw_graph, graph_split.first);
  OpenDataflowSubgraphResult subgraph_res2 = get_subgraph(
      sub_pcg_from_full_pcg(context.pcg).raw_graph, graph_split.second);

  std::unordered_set<DataflowOutput> split_outputs = transform(
      keys(subgraph_res2.full_graph_values_to_subgraph_inputs),
      [](OpenDataflowValue const &v) { return v.get<DataflowOutput>(); });

  for (std::unordered_map<DataflowOutput, MachineView> const
           &split_machine_views :
       allowed_machine_mappings(context, split_outputs, resource)) {
    std::unordered_map<OpenDataflowValue, MachineView> fixed_machine_views1 =
        restrict_keys(fixed_machine_views,
                      get_open_dataflow_values(subgraph_res1.graph));
    std::unordered_map<OpenDataflowValue, MachineView> fixed_machine_views2 =
        restrict_keys(fixed_machine_views,
                      get_open_dataflow_values(subgraph_res2.graph));

    for (auto const &[full_graph_value, subgraph_input] :
         subgraph_res2.full_graph_values_to_subgraph_inputs) {
      MachineView mv =
          split_machine_views.at(full_graph_value.get<DataflowOutput>());
      fixed_machine_views1.emplace(full_graph_value, mv);
      fixed_machine_views2.emplace(OpenDataflowValue(subgraph_input), mv);
    }

    minimize_runtime(
        optimal_result,
        sequential_combine(
            get_optimal_machine_mapping_internal(
                context, decompn1, resource, fixed_machine_views1),
            get_optimal_machine_mapping_internal(
                context, decompn2, resource, fixed_machine_views2)));
  }

  return optimal_result;
}

MachineMappingResult get_optimal_machine_mapping_internal(
    MachineMappingContext &context,
    ParallelSplit const &parallel,
    MachineSpecification const &resource,
    std::unordered_map<OpenDataflowValue, MachineView> const
        &fixed_machine_views) {
  auto [decompn1, decompn2] = split_sp_decomposition(parallel);

  GraphSplit graph_split = get_graph_split(decompn1, decompn2);

  OpenDataflowSubgraphResult subgraph_res1 = get_subgraph(
      sub_pcg_from_full_pcg(context.pcg).raw_graph, graph_split.first);
  OpenDataflowSubgraphResult subgraph_res2 = get_subgraph(
      sub_pcg_from_full_pcg(context.pcg).raw_graph, graph_split.second);

  std::unordered_map<OpenDataflowValue, MachineView> fixed_machine_views1 =
      restrict_keys(fixed_machine_views,
                    get_open_dataflow_values(subgraph_res1.graph));
  std::unordered_map<OpenDataflowValue, MachineView> fixed_machine_views2 =
      restrict_keys(fixed_machine_views,
                    get_open_dataflow_values(subgraph_res2.graph));

  MachineMappingResult optimal_result = sequential_combine(
      get_optimal_machine_mapping_internal(
          context, decompn1, resource, fixed_machine_views1),
      get_optimal_machine_mapping_internal(
          context, decompn2, resource, fixed_machine_views2));

  for (auto const &resource_split : get_resource_split(resource)) {
    minimize_runtime(
        optimal_result,
        parallel_combine(
            get_optimal_machine_mapping_internal(
                context, decompn1, resource_split.first, fixed_machine_views1),
            get_optimal_machine_mapping_internal(context,
                                                 decompn2,
                                                 resource_split.second,
                                                 fixed_machine_views2)));
  }

  return optimal_result;
}

MachineMappingResult get_optimal_machine_mapping_internal(
    MachineMappingContext &context,
    Node const &node,
    MachineSpecification const &resource,
    std::unordered_map<OpenDataflowValue, MachineView> const
        &fixed_machine_views) {

  SubParallelComputationGraph subgraph = get_pcg_subgraph(context.pcg, {node});

  OpenDataflowValue any_output =
      OpenDataflowValue(get_outputs(context.pcg.raw_graph, node)[0]);
  if (contains_key(fixed_machine_views, any_output)) {
    {
      std::unordered_set<MachineView> allowed_machine_views_for_node = context.allowed_machine_views(
          context.pcg.raw_graph.at(node), resource);
      MachineView fixed_machine_view_for_node = fixed_machine_views.at(any_output);
      assert(contains(allowed_machine_views_for_node, fixed_machine_view_for_node));
    }
    MachineView mv = fixed_machine_views.at(any_output);
    MachineMapping mv_map{{{node, mv}}};
    return MachineMappingResult(base_case_estimate_cost(subgraph,
                                                        context.cost_estimator,
                                                        fixed_machine_views),
                                mv_map);
  } else {
    MachineMappingResult optimal_result = get_infinity_machine_mapping_result();
    for (std::unordered_map<Node, MachineView> node_machine_views :
         allowed_machine_mappings(context, {node}, resource)) {
      MachineView mv = node_machine_views.at(node);
      MachineMapping mv_map{{{node, mv}}};

      std::vector<OpenDataflowValue> outputs_of_node = transform(
          get_outputs(context.pcg.raw_graph, node),
          [](DataflowOutput const &o) { return OpenDataflowValue(o); });

      std::unordered_map<OpenDataflowValue, MachineView> output_mv_map =
          generate_map(outputs_of_node,
                       [&](OpenDataflowValue const &o) { return mv; });

      std::unordered_map<OpenDataflowValue, MachineView> machine_views =
          merge_maps(fixed_machine_views, output_mv_map);
      minimize_runtime(optimal_result,
                       MachineMappingResult(
                           base_case_estimate_cost(
                               subgraph, context.cost_estimator, machine_views),
                           mv_map));
    }
    return optimal_result;
  }
}

} // namespace FlexFlow
