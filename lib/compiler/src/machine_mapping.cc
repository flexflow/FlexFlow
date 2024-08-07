#include "compiler/machine_mapping.h"
#include "compiler/cost_estimator.h"
#include "pcg/machine_specification.dtg.h"
#include "pcg/machine_specification.h"
#include "pcg/machine_view.dtg.h"
#include "pcg/machine_view.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "utils/containers.h"
#include "utils/containers/are_disjoint.h"
#include "utils/containers/as_vector.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/get_first.h"
#include "utils/containers/get_only.h"
#include "utils/containers/keys.h"
#include "utils/containers/restrict_keys.h"
#include "utils/containers/set_minus.h"
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

namespace FlexFlow {

MachineMapping combine(MachineMapping const &s1, MachineMapping const &s2) {
  return MachineMapping{merge_maps(s1.machine_views, s2.machine_views)};
}

bool nodes_are_disjoint(MachineMapping const &m1, MachineMapping const &m2) {
  return are_disjoint(keys(m1.machine_views), keys(m2.machine_views));
}

void minimize_runtime(OptimalCostResult &m1, OptimalCostResult const &m2) {
  if (m2.runtime < m1.runtime) {
    m1 = m2;
  }
}

OptimalCostResult
    OptimalCostResult::sequential_combine(OptimalCostResult const &s1,
                                          OptimalCostResult const &s2) {
  return OptimalCostResult{s1.runtime + s2.runtime,
                           combine(s1.machine_mapping, s2.machine_mapping)};
}

OptimalCostResult
    OptimalCostResult::parallel_combine(OptimalCostResult const &s1,
                                        OptimalCostResult const &s2) {
  return OptimalCostResult{std::max(s1.runtime, s2.runtime),
                           combine(s1.machine_mapping, s2.machine_mapping)};
}

OptimalCostResult OptimalCostResult::infinity() {
  return {std::numeric_limits<float>::infinity(),
          MachineMapping{std::unordered_map<Node, MachineView>{}}};
}

std::optional<OptimalCostResult>
    OptimalCostCache::load(OptimalCostState const &state) const {
  if (contains_key(cache, state)) {
    OptimalCostResult result = cache.at(state);
    return std::make_optional(result);
  }
  return std::nullopt;
}

void OptimalCostCache::save(OptimalCostState const &state,
                            OptimalCostResult const &result) {
  assert(!contains_key(cache, state));
  cache.emplace(state, result);
}

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

std::pair<SerialParallelDecomposition, SerialParallelDecomposition>
    decompose(SerialSplit const &serial) {
  if (serial.children.size() == 2) {
    return {widen<SerialParallelDecomposition>(serial.children[0]),
            widen<SerialParallelDecomposition>(serial.children[1])};
  }
  SerialSplit decompn1 = serial;
  decompn1.children.pop_back();
  return {SerialParallelDecomposition(decompn1),
          widen<SerialParallelDecomposition>(serial.children.back())};
}

std::pair<SerialParallelDecomposition, SerialParallelDecomposition>
    decompose(ParallelSplit const &parallel) {
  if (parallel.children.size() == 2) {
    std::vector<SerialParallelDecomposition> children =
        transform(as_vector(parallel.children), [&](auto const &child) {
          return widen<SerialParallelDecomposition>(child);
        });
    return {children[0], children[1]};
  }
  ParallelSplit decompn1 = parallel;
  std::variant<SerialSplit, Node> child = *parallel.children.begin();
  decompn1.children.erase(child);
  return {SerialParallelDecomposition(decompn1),
          widen<SerialParallelDecomposition>(child)};
}

GraphSplit
    get_graph_split(SerialParallelDecomposition const &pre_decomposition,
                    SerialParallelDecomposition const &post_decomposition) {
  return GraphSplit{get_nodes(pre_decomposition),
                    get_nodes(post_decomposition)};
}

float estimate_cost(
    SubParallelComputationGraph const &g,
    CostEstimator const &estimator,
    std::unordered_map<OpenDataflowValue, MachineView> const &machine_views) {
  // TODO: Consider parallelism
  float cost = 1.;
  // for (Node const &node : get_nodes(g.raw_graph)) {
  //   std::vector<OpenDataflowEdge> incoming_edges =
  //       get_incoming_edges(g.raw_graph, node);
  //   std::vector<ParallelTensorShape> inputs =
  //       transform(incoming_edges,
  //                 [&](OpenDataflowEdge const &input_edge) {
  //                   return g.raw_graph.at(input_edge).get_shape();
  //                 });
  //   cost += estimator.estimate_cost(
  //       g.raw_graph.at(node).op_attrs, inputs,
  //       device_mapping.machine_views.at(node));
  // }
  return cost;
}

struct MachineMappingSearcher {
  MachineMappingSearcher(
      ParallelComputationGraph const &pcg,
      CostEstimator const &cost_estimator,
      std::function<std::unordered_set<MachineView>(
          ParallelLayerAttrs const &, MachineSpecification const &)> const
          &allowed_machine_views,
      OptimalCostCache &cached_subgraph_costs)
      : pcg(pcg), cost_estimator(cost_estimator),
        allowed_machine_views(allowed_machine_views),
        cached_subgraph_costs(cached_subgraph_costs) {}

  ParallelComputationGraph pcg;
  CostEstimator cost_estimator;
  std::function<std::unordered_set<MachineView>(ParallelLayerAttrs const &,
                                                MachineSpecification const &)>
      allowed_machine_views;
  OptimalCostCache &cached_subgraph_costs;

  struct OptimalCostFunctor {
    OptimalCostFunctor(
        MachineMappingSearcher *searcher,
        MachineSpecification resource,
        std::unordered_map<OpenDataflowValue, MachineView> fixed_machine_views)
        : searcher(searcher), resource(resource),
          fixed_machine_views(fixed_machine_views) {}

    MachineMappingSearcher *searcher;
    MachineSpecification resource;
    std::unordered_map<OpenDataflowValue, MachineView> fixed_machine_views;

    template <typename T>
    OptimalCostResult operator()(T const &t) {
      OptimalCostState state(
          SerialParallelDecomposition{t}, resource, fixed_machine_views);
      std::optional<OptimalCostResult> cached_result =
          searcher->cached_subgraph_costs.load(state);

      if (cached_result) {
        return cached_result.value();
      }
      OptimalCostResult result =
          searcher->optimal_cost(t, resource, fixed_machine_views);

      searcher->cached_subgraph_costs.save(state, result);
      return result;
    }
  };

  OptimalCostResult optimal_cost(MachineSpecification resource) {
    return std::visit(
        OptimalCostFunctor(this, resource, {}),
        get_serial_parallel_decomposition(pcg.raw_graph).value().raw_variant);
  }

  OptimalCostResult
      optimal_cost(SerialSplit const &serial,
                   MachineSpecification const &resource,
                   std::unordered_map<OpenDataflowValue, MachineView> const
                       &fixed_machine_views) {
    OptimalCostResult optimal_result = OptimalCostResult::infinity();

    auto [decompn1, decompn2] = decompose(serial);

    GraphSplit graph_split = get_graph_split(decompn1, decompn2);

    OpenDataflowSubgraphResult subgraph_res1 =
        get_subgraph(sub_pcg_from_full_pcg(pcg).raw_graph, graph_split.first);
    OpenDataflowSubgraphResult subgraph_res2 =
        get_subgraph(sub_pcg_from_full_pcg(pcg).raw_graph, graph_split.second);

    std::unordered_set<DataflowOutput> split_outputs;
    for (auto const &[value, _] :
         subgraph_res2.full_graph_values_to_subgraph_inputs) {
      assert(value.has<DataflowOutput>());
      split_outputs.insert(value.get<DataflowOutput>());
    }

    for (std::unordered_map<DataflowOutput, MachineView> const
             &split_machine_views :
         enumerate_machine_views(split_outputs, resource)) {
      std::unordered_map<OpenDataflowValue, MachineView> fixed_machine_views1 =
          restrict_keys(fixed_machine_views,
                        get_open_dataflow_values(subgraph_res1.graph));
      std::unordered_map<OpenDataflowValue, MachineView> fixed_machine_views2 =
          restrict_keys(fixed_machine_views,
                        get_open_dataflow_values(subgraph_res2.graph));

      for (auto const &[split_value, split_input] :
           subgraph_res2.full_graph_values_to_subgraph_inputs) {
        MachineView mv =
            split_machine_views.at(split_value.get<DataflowOutput>());
        fixed_machine_views1.emplace(split_value, mv);
        fixed_machine_views2.emplace(OpenDataflowValue(split_input), mv);
      }

      minimize_runtime(optimal_result,
                       OptimalCostResult::sequential_combine(
                           std::visit(OptimalCostFunctor(
                                          this, resource, fixed_machine_views1),
                                      decompn1.raw_variant),
                           std::visit(OptimalCostFunctor(
                                          this, resource, fixed_machine_views2),
                                      decompn2.raw_variant)));
    }

    return optimal_result;
  }

  OptimalCostResult
      optimal_cost(ParallelSplit const &parallel,
                   MachineSpecification const &resource,
                   std::unordered_map<OpenDataflowValue, MachineView> const
                       &fixed_machine_views) {
    auto [decompn1, decompn2] = decompose(parallel);

    GraphSplit graph_split = get_graph_split(decompn1, decompn2);

    OpenDataflowSubgraphResult subgraph_res1 =
        get_subgraph(sub_pcg_from_full_pcg(pcg).raw_graph, graph_split.first);
    OpenDataflowSubgraphResult subgraph_res2 =
        get_subgraph(sub_pcg_from_full_pcg(pcg).raw_graph, graph_split.second);

    std::unordered_map<OpenDataflowValue, MachineView> fixed_machine_views1 =
        restrict_keys(fixed_machine_views,
                      get_open_dataflow_values(subgraph_res1.graph));
    std::unordered_map<OpenDataflowValue, MachineView> fixed_machine_views2 =
        restrict_keys(fixed_machine_views,
                      get_open_dataflow_values(subgraph_res2.graph));

    OptimalCostResult optimal_result = OptimalCostResult::sequential_combine(
        std::visit(OptimalCostFunctor(this, resource, fixed_machine_views1),
                   decompn1.raw_variant),
        std::visit(OptimalCostFunctor(this, resource, fixed_machine_views1),
                   decompn2.raw_variant));

    for (auto const &resource_split : get_resource_split(resource)) {
      minimize_runtime(
          optimal_result,
          OptimalCostResult::parallel_combine(
              std::visit(OptimalCostFunctor(
                             this, resource_split.first, fixed_machine_views1),
                         decompn1.raw_variant),
              std::visit(OptimalCostFunctor(
                             this, resource_split.second, fixed_machine_views1),
                         decompn2.raw_variant)));
    }

    return optimal_result;
  }

  OptimalCostResult
      optimal_cost(Node const &node,
                   MachineSpecification const &resource,
                   std::unordered_map<OpenDataflowValue, MachineView> const
                       &fixed_machine_views) {
    SubParallelComputationGraph subgraph =
        sub_pcg_from_partial_pcg(pcg, {node});

    OpenDataflowValue any_output =
        OpenDataflowValue(get_outputs(pcg.raw_graph, node)[0]);
    if (contains_key(fixed_machine_views, any_output)) {
      assert(contains(allowed_machine_views(pcg.raw_graph.at(node), resource),
                      fixed_machine_views.at(any_output)));
      MachineView mv = fixed_machine_views.at(any_output);
      MachineMapping mv_map{{{node, mv}}};
      return {estimate_cost(subgraph, cost_estimator, fixed_machine_views),
              mv_map};
    } else {
      OptimalCostResult optimal_result = OptimalCostResult::infinity();
      for (std::unordered_map<Node, MachineView> node_machine_views :
           enumerate_machine_views({node}, resource)) {
        MachineMapping mv_map{{{node, node_machine_views.at(node)}}};
        std::unordered_map<OpenDataflowValue, MachineView> machine_views =
            fixed_machine_views;
        for (DataflowOutput o : get_outputs(pcg.raw_graph, node)) {
          machine_views.emplace(o, node_machine_views.at(node));
        }
        minimize_runtime(
            optimal_result,
            {estimate_cost(subgraph, cost_estimator, machine_views), mv_map});
      }
      return optimal_result;
    }
  }

  std::vector<std::unordered_map<Node, MachineView>>
      enumerate_machine_views(std::unordered_set<Node> const &nodes,
                              MachineSpecification const &resource) {
    if (nodes.empty()) {
      return {{}};
    }
    Node node = get_first(nodes);
    std::vector<std::unordered_map<Node, MachineView>> partial_enumeration =
        enumerate_machine_views(set_minus(nodes, {node}), resource);
    std::unordered_set<MachineView> allowed_machine_views_for_node =
        this->allowed_machine_views(pcg.raw_graph.at(node), resource);
    std::vector<std::unordered_map<Node, MachineView>> enumeration;
    for (MachineView const &mv : allowed_machine_views_for_node) {
      for (std::unordered_map<Node, MachineView> const &partial :
           partial_enumeration) {
        enumeration.push_back(merge_maps(
            partial, std::unordered_map<Node, MachineView>{{node, mv}}));
      }
    }
    return enumeration;
  }

  std::vector<std::unordered_map<DataflowOutput, MachineView>>
      enumerate_machine_views(std::unordered_set<DataflowOutput> const &values,
                              MachineSpecification const &resource) {
    std::unordered_set<Node> nodes;
    for (DataflowOutput const &v : values) {
      nodes.insert(v.node);
    }

    std::vector<std::unordered_map<Node, MachineView>> node_enumeration =
        enumerate_machine_views(nodes, resource);
    std::vector<std::unordered_map<DataflowOutput, MachineView>> enumeration;

    for (std::unordered_map<Node, MachineView> _node_enumeration :
         node_enumeration) {
      std::unordered_map<DataflowOutput, MachineView> _emumeration;
      for (DataflowOutput const &v : values) {
        _emumeration.emplace(v, _node_enumeration.at(v.node));
      }
      enumeration.push_back(_emumeration);
    }

    return enumeration;
  }
};

OptimalCostResult optimal_cost(
    ParallelComputationGraph const &g,
    std::function<std::unordered_set<MachineView>(
        ParallelLayerAttrs const &, MachineSpecification const &)> const
        &allowed_machine_views,
    CostEstimator const &cost_estimator,
    MachineSpecification const &resources,
    OptimalCostCache &cached_subgraph_costs) {
  MachineMappingSearcher searcher(
      g, cost_estimator, allowed_machine_views, cached_subgraph_costs);
  return searcher.optimal_cost(resources);
}

} // namespace FlexFlow
