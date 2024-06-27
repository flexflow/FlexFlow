#include "compiler/machine_mapping.h"
#include "compiler/cost_estimate.h"
#include "graph_utils.h"
#include "pcg/parallel_computation_graph.h"
#include "utils/exception.h"
#include "utils/graph/serialparallel.h"

#include "utils/deduplicated_priority_queue.h"
#include <algorithm>

namespace FlexFlow {

MachineMapping MachineMapping::combine(MachineMapping const &s1,
                                       MachineMapping const &s2) {
  return MachineMapping{merge_maps(s1.machine_views, s2.machine_views)};
}

bool MachineMapping::nodes_are_disjoint(MachineMapping const &m1,
                                        MachineMapping const &m2) {
  return are_disjoint(keys(m1.machine_views), keys(m2.machine_views));
}

OptimalCostResult
    OptimalCostResult::sequential_combine(OptimalCostResult const &s1,
                                          OptimalCostResult const &s2) {
  return OptimalCostResult{
      s1.runtime + s2.runtime,
      MachineMapping::combine(s1.machine_mapping, s2.machine_mapping)};
}

OptimalCostResult
    OptimalCostResult::parallel_combine(OptimalCostResult const &s1,
                                        OptimalCostResult const &s2) {
  return OptimalCostResult{
      std::max(s1.runtime, s2.runtime),
      MachineMapping::combine(s1.machine_mapping, s2.machine_mapping)};
}

OptimalCostResult OptimalCostResult::infinity() {
  return {std::numeric_limits<float>::infinity(),
          MachineMapping{std::unordered_map<Node, MachineView>{}}};
}

bool OptimalCostRuntimeCmp::operator()(OptimalCostResult const &lhs,
                                       OptimalCostResult const &rhs) {
  return lhs.runtime < rhs.runtime;
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

// We may replace this by having unflattened AST
template <typename T>
std::pair<SerialParallelDecomposition, SerialParallelDecomposition>
    decompose(T const &t) {
  if (t.children.size() == 2) {
    return {widen<SerialParallelDecomposition>(t.children[0]),
            widen<SerialParallelDecomposition>(t.children[1])};
  }
  T decompn1 = t;
  decompn1.children.pop_back();
  return {decompn1, widen<SerialParallelDecomposition>(t.children.back())};
}

GraphSplit
    get_graph_split(SerialParallelDecomposition const &pre_decomposition,
                    SerialParallelDecomposition const &post_decomposition) {
  return {get_nodes(pre_decomposition), get_nodes(post_decomposition)};
}

float estimate_cost(SubParallelComputationGraphView const &g,
                    CostEstimator const &estimator,
                    MachineMapping const &device_mapping,
                    std::unordered_map<OpenMultiDiEdge, MachineView> const
                        &frontier_machine_views) {
  // TODO: Consider parallelism
  float cost = 0;
  for (Node const &node : get_nodes(g)) {
    std::unordered_set<UpwardOpenMultiDiEdge> incoming_edges =
        get_incoming_edges(g, node);
    std::vector<ParallelTensorShape> inputs =
        transform(as_vector(incoming_edges),
                  [&](UpwardOpenMultiDiEdge const &input_edge) {
                    return g.at(input_edge).get_shape();
                  });
    cost += estimator.estimate_cost(
        g.at(node).attrs, inputs, device_mapping.machine_views.at(node));
  }
  return cost;
}

void minimize_runtime(OptimalCostResult &m1, OptimalCostResult const &m2) {
  minimize(m1, m2, OptimalCostRuntimeCmp{});
}

struct MachineMappingSearcher {
  MachineMappingSearcher(
      CostEstimator cost_estimator,
      std::function<std::unordered_set<MachineView>(
          Operator const &, MachineSpecification const &)> const
          &allowed_machine_views,
      OptimalCostCache &cached_subgraph_costs)
      : cost_estimator(cost_estimator),
        allowed_machine_views(allowed_machine_views),
        cached_subgraph_costs(cached_subgraph_costs) {}

  CostEstimator cost_estimator;
  std::function<std::unordered_set<MachineView>(Operator const &,
                                                MachineSpecification const &)>
      allowed_machine_views;
  OptimalCostCache &cached_subgraph_costs;

  struct OptimalCostFunctor {
    OptimalCostFunctor(
        MachineMappingSearcher *searcher,
        SubParallelComputationGraphView const &g,
        MachineSpecification resource,
        std::unordered_map<Node, MachineView> given_machine_views,
        std::unordered_map<OpenMultiDiEdge, MachineView> frontier_machine_views)
        : searcher(searcher), g(g), resource(resource),
          given_machine_views(given_machine_views),
          frontier_machine_views(frontier_machine_views) {}

    MachineMappingSearcher *searcher;
    SubParallelComputationGraphView const &g;
    MachineSpecification resource;
    std::unordered_map<Node, MachineView> given_machine_views;
    std::unordered_map<OpenMultiDiEdge, MachineView> frontier_machine_views;

    template <typename T>
    OptimalCostResult operator()(T const &t) {
      OptimalCostState state{
          t, resource, given_machine_views, frontier_machine_views};
      std::optional<OptimalCostResult> cached_result =
          searcher->cached_subgraph_costs.load(state);

      if (cached_result) {
        return cached_result.value();
      }
      OptimalCostResult result = searcher->optimal_cost(
          t, g, resource, given_machine_views, frontier_machine_views);

      searcher->cached_subgraph_costs.save(state, result);
      return result;
    }
  };

  OptimalCostResult
      optimal_cost(SubParallelComputationGraphView const &g,
                   MachineSpecification resource,
                   SerialParallelDecomposition const &sp_decomposition) {
    return visit(OptimalCostFunctor(this, g, resource, {}, {}),
                 sp_decomposition);
  }

  OptimalCostResult optimal_cost(
      Serial const &serial,
      SubParallelComputationGraphView const &g,
      MachineSpecification const &resource,
      std::unordered_map<Node, MachineView> const &given_machine_views,
      std::unordered_map<OpenMultiDiEdge, MachineView> const
          &frontier_machine_views) {

    auto decomposed = decompose(serial);
    SerialParallelDecomposition pre_decompn = decomposed.first;
    SerialParallelDecomposition post_decompn = decomposed.second;

    GraphSplit graph_split = get_graph_split(pre_decompn, post_decompn);
    SubParallelComputationGraphView pre_graph =
        get_subgraph<OpenMultiDiSubgraphView>(g, graph_split.first);
    SubParallelComputationGraphView post_graph =
        get_subgraph<DownwardOpenMultiDiSubgraphView>(g, graph_split.second);

    std::unordered_set<Node> post_graph_sources =
        get_closed_sources(post_graph);

    assert(post_graph_sources.size() == 1); // assume perfect SP

    Node split_point = get_only(post_graph_sources);
    OutputMultiDiEdge split_edge = get_only(get_open_outputs(pre_graph));

    OptimalCostResult optimal_result = OptimalCostResult::infinity();

    for (MachineView const &mv :
         allowed_machine_views(g.at(split_point), resource)) {
      std::unordered_map<Node, MachineView> new_given_machine_views =
          given_machine_views;
      new_given_machine_views.emplace(split_point, mv);
      std::unordered_map<OpenMultiDiEdge, MachineView>
          new_frontier_machine_views = frontier_machine_views;
      new_frontier_machine_views.emplace(split_edge, mv);
      minimize_runtime(optimal_result,
                       OptimalCostResult::sequential_combine(
                           visit(OptimalCostFunctor(this,
                                                    pre_graph,
                                                    resource,
                                                    given_machine_views,
                                                    new_frontier_machine_views),
                                 pre_decompn),
                           visit(OptimalCostFunctor(this,
                                                    post_graph,
                                                    resource,
                                                    new_given_machine_views,
                                                    frontier_machine_views),
                                 post_decompn)));
    }

    return optimal_result;
  }

  OptimalCostResult optimal_cost(
      Parallel const &parallel,
      SubParallelComputationGraphView const &g,
      MachineSpecification const &resource,
      std::unordered_map<Node, MachineView> const &given_machine_views,
      std::unordered_map<OpenMultiDiEdge, MachineView> const
          &frontier_machine_views) {
    auto decomposed = decompose(parallel);
    SerialParallelDecomposition decompn1 = decomposed.first;
    SerialParallelDecomposition decompn2 = decomposed.second;

    GraphSplit graph_split = get_graph_split(decompn1, decompn2);
    SubParallelComputationGraphView g1 = get_subgraph<OpenMultiDiSubgraphView>(
                                        g, graph_split.first),
                                    g2 = get_subgraph<OpenMultiDiSubgraphView>(
                                        g, graph_split.second);

    OptimalCostResult optimal_result = OptimalCostResult::sequential_combine(
        visit(OptimalCostFunctor(this,
                                 g1,
                                 resource,
                                 given_machine_views,
                                 frontier_machine_views),
              decompn1),
        visit(OptimalCostFunctor(this,
                                 g2,
                                 resource,
                                 given_machine_views,
                                 frontier_machine_views),
              decompn2));

    for (auto const &resource_split : get_resource_split(resource)) {
      minimize_runtime(optimal_result,
                       OptimalCostResult::parallel_combine(
                           visit(OptimalCostFunctor(this,
                                                    g1,
                                                    resource_split.first,
                                                    given_machine_views,
                                                    frontier_machine_views),
                                 decompn1),
                           visit(OptimalCostFunctor(this,
                                                    g2,
                                                    resource_split.second,
                                                    given_machine_views,
                                                    frontier_machine_views),
                                 decompn2)));
    }

    return optimal_result;
  }

  OptimalCostResult optimal_cost(
      Node const &node,
      SubParallelComputationGraphView const &g,
      MachineSpecification const &resource,
      std::unordered_map<Node, MachineView> const &given_machine_views,
      std::unordered_map<OpenMultiDiEdge, MachineView> const
          &frontier_machine_views) {
    if (contains_key(given_machine_views, node)) {
      assert(contains(allowed_machine_views(g.at(node), resource),
                      given_machine_views.at(node)));
      MachineMapping mv_map{given_machine_views};
      return {estimate_cost(g, cost_estimator, mv_map, frontier_machine_views),
              mv_map};
    } else {
      OptimalCostResult optimal_result = OptimalCostResult::infinity();
      for (auto mv : allowed_machine_views(g.at(node), resource)) {
        MachineMapping mv_map{{{node, mv}}};
        minimize_runtime(
            optimal_result,
            {estimate_cost(g, cost_estimator, mv_map, frontier_machine_views),
             mv_map});
      }
      return optimal_result;
    }
  }
};

OptimalCostResult
    optimal_cost(ParallelComputationGraph const &g,
                 std::function<std::unordered_set<MachineView>(
                     Operator const &, MachineSpecification const &)> const
                     &allowed_machine_views,
                 CostEstimator const &cost_estimator,
                 MachineSpecification const &resources,
                 OptimalCostCache &cached_subgraph_costs) {
  SerialParallelDecomposition sp_decomposition =
      get_serial_parallel_decomposition(g);
  SubParallelComputationGraphView subpcg = pcg_to_subpcg(g);
  MachineMappingSearcher searcher(
      cost_estimator, allowed_machine_views, cached_subgraph_costs);
  return searcher.optimal_cost(subpcg, resources, sp_decomposition);
}

} // namespace FlexFlow
