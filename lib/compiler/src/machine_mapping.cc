#include "compiler/machine_mapping.h"
#include "compiler/cost_estimate.h"
#include "graph_utils.h"
#include "pcg/parallel_computation_graph.h"
#include "utils/exception.h"
#include "utils/graph/serialparallel.h"

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

optional<OptimalCostResult>
    OptimalCostCache::load(OptimalCostState const &state) const {
  if (contains_key(cache, state)) {
    return make_optional(cache.at(state));
  }
  return nullopt;
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

std::pair<SubParallelComputationGraph, SubParallelComputationGraph>
    apply_split(SubParallelComputationGraph const &g, GraphSplit const &split) {
  OpenMultiDiGraphView g1 = get_subgraph(g, split.first);
  OpenMultiDiGraphView g2 = get_subgraph(g, split.second);

  if (get_edge_splits(g, split).size() > 0) {
    // Sequential split
    if (get_open_sinks(g1).size() <= get_open_sources(g2).size()) {
      // get_open_sinks(*g1).size() should be 1 in perfect sp graphs
      return {get_subgraph<OpenType::UPWARD>(g, split.first),
              get_subgraph<OpenType::CLOSED>(g, split.second)};
    } else {
      return {get_subgraph<OpenType::OPEN>(g, split.first),
              get_subgraph<OpenType::DOWNWARD>(g, split.first)};
    }
  } else {
    // Parallel split
    return {get_subgraph<OpenType::OPEN>(g, split.first),
            get_subgraph<OpenType::OPEN>(g, split.second)};
  }
}

float estimate_cost(SubParallelComputationGraph const &g,
                    CostEstimator const &estimator,
                    MachineMapping const &device_mapping) {
  NOT_IMPLEMENTED();
}

void minimize_runtime(OptimalCostResult &m1, OptimalCostResult const &m2) {
  minimize(m1, m2, OptimalCostRuntimeCmp{});
}

struct OptimalCost {
  OptimalCost(
      SubParallelComputationGraph const &g,
      CostEstimator const &cost_estimator,
      MachineSpecification const &resource,
      optional<MachineView> const &source_machine_view, // assume perfect SP
      optional<MachineView> const &sink_machine_view,
      std::function<std::unordered_set<MachineView>(
          Operator const &, MachineSpecification const &)> const
          &allowed_machine_views,
      OptimalCostCache &cached_subgraph_costs)
      : g(g), cost_estimator(cost_estimator), resource(resource),
        source_machine_view(source_machine_view),
        sink_machine_view(sink_machine_view),
        allowed_machine_views(allowed_machine_views),
        cached_subgraph_costs(cached_subgraph_costs) {}

  SubParallelComputationGraph const &g;
  CostEstimator const &cost_estimator;
  MachineSpecification const &resource;
  optional<MachineView> const &source_machine_view;
  optional<MachineView> const &sink_machine_view;
  std::function<std::unordered_set<MachineView>(
      Operator const &, MachineSpecification const &)> const
      &allowed_machine_views;
  OptimalCostCache &cached_subgraph_costs;

  template <typename T>
  OptimalCostResult operator()(T const &t) const {
    OptimalCostState state{g, resource, source_machine_view, sink_machine_view};
    optional<OptimalCostResult> cached_result =
        cached_subgraph_costs.load(state);

    if (cached_result) {
      return cached_result.value();
    }

    OptimalCostResult result = this->optimal_cost(t);

    cached_subgraph_costs.save(state, result);
    return result;
  }

  OptimalCostResult optimal_cost(Serial const &serial) const {
    auto decomposed = decompose(serial);
    SerialParallelDecomposition pre_decompn = decomposed.first;
    SerialParallelDecomposition post_decompn = decomposed.second;

    auto subgraphs = apply_split(g, get_graph_split(pre_decompn, post_decompn));
    SubParallelComputationGraph pre_graph = subgraphs.first,
                                post_graph = subgraphs.second;

    std::unordered_set<Node> pre_graph_sinks = get_closed_sinks(pre_graph);
    std::unordered_set<Node> post_graph_sources =
        get_closed_sources(post_graph);

    assert(pre_graph_sinks.size() + post_graph_sources.size() ==
           1); // assume perfect SP

    Node const &split_point =
        get_only(set_union(pre_graph_sinks, post_graph_sources));

    OptimalCostResult optimal_result = OptimalCostResult::infinity();

    for (MachineView const &mv :
         allowed_machine_views(g.at(split_point), resource)) {
      optional<MachineView> pre_sink_mv =
          contains(pre_graph_sinks, split_point) ? make_optional(mv) : nullopt;
      optional<MachineView> post_source_mv =
          contains(post_graph_sources, split_point) ? make_optional(mv)
                                                    : nullopt;
      minimize_runtime(optimal_result,
                       OptimalCostResult::sequential_combine(
                           visit(OptimalCost(pre_graph,
                                             cost_estimator,
                                             resource,
                                             source_machine_view,
                                             pre_sink_mv,
                                             allowed_machine_views,
                                             cached_subgraph_costs),
                                 pre_decompn),
                           visit(OptimalCost(post_graph,
                                             cost_estimator,
                                             resource,
                                             post_source_mv,
                                             sink_machine_view,
                                             allowed_machine_views,
                                             cached_subgraph_costs),
                                 post_decompn)));
    }

    return optimal_result;
  }

  OptimalCostResult optimal_cost(Parallel const &parallel) const {
    auto decomposed = decompose(parallel);
    SerialParallelDecomposition decompn1 = decomposed.first;
    SerialParallelDecomposition decompn2 = decomposed.second;

    auto subgraphs = apply_split(g, get_graph_split(decompn1, decompn2));
    SubParallelComputationGraph g1 = subgraphs.first, g2 = subgraphs.second;

    OptimalCostResult optimal_result = OptimalCostResult::sequential_combine(
        visit(OptimalCost(g1,
                          cost_estimator,
                          resource,
                          source_machine_view,
                          sink_machine_view,
                          allowed_machine_views,
                          cached_subgraph_costs),
              decompn1),
        visit(OptimalCost(g2,
                          cost_estimator,
                          resource,
                          source_machine_view,
                          sink_machine_view,
                          allowed_machine_views,
                          cached_subgraph_costs),
              decompn2));

    for (auto const &resource_split : get_resource_split(resource)) {
      minimize_runtime(optimal_result,
                       OptimalCostResult::parallel_combine(
                           visit(OptimalCost(g1,
                                             cost_estimator,
                                             resource_split.first,
                                             source_machine_view,
                                             sink_machine_view,
                                             allowed_machine_views,
                                             cached_subgraph_costs),
                                 decompn1),
                           visit(OptimalCost(g2,
                                             cost_estimator,
                                             resource_split.second,
                                             source_machine_view,
                                             sink_machine_view,
                                             allowed_machine_views,
                                             cached_subgraph_costs),
                                 decompn2)));
    }

    return optimal_result;
  }

  OptimalCostResult optimal_cost(Node const &node) const {
    if (source_machine_view) {
      assert(get_closed_sources(g).empty());
      assert(contains(allowed_machine_views(g.at(node), resource),
                      source_machine_view.value()));
      MachineMapping mv_map{{{node, source_machine_view.value()}}};
      return {estimate_cost(g, cost_estimator, mv_map), mv_map};
    } else if (sink_machine_view) {
      assert(get_closed_sinks(g).empty());
      assert(contains(allowed_machine_views(g.at(node), resource),
                      sink_machine_view.value()));
      MachineMapping mv_map{{{node, sink_machine_view.value()}}};
      return {estimate_cost(g, cost_estimator, mv_map), mv_map};
    } else {
      OptimalCostResult optimal_result = OptimalCostResult::infinity();
      for (auto mv : allowed_machine_views(g.at(node), resource)) {
        MachineMapping mv_map{{{node, mv}}};
        minimize_runtime(optimal_result,
                         {estimate_cost(g, cost_estimator, mv_map), mv_map});
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
  return visit(OptimalCost(pcg_to_subpcg(g),
                           cost_estimator,
                           resources,
                           nullopt,
                           nullopt,
                           allowed_machine_views,
                           cached_subgraph_costs),
               get_serial_parallel_decomposition(g));
}

} // namespace FlexFlow
