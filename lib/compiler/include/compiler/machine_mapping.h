#ifndef _FLEXFLOW_COMPILER_MACHINE_MAPPING_H
#define _FLEXFLOW_COMPILER_MACHINE_MAPPING_H

#include "cost_estimate.h"
#include "pcg/machine_specification.h"
#include "pcg/machine_view.h"
#include "pcg/parallel_computation_graph.h"
#include "substitutions/sub_parallel_computation_graph.h"

namespace FlexFlow {

using SubParallelComputationGraphView =
    OutputLabelledOpenMultiDiGraphView<Operator, ParallelTensor>;

enum class OpMemoryDecision {
  None,
  Remat,
};

template <typename T>
struct NodeMapping {
  static NodeMapping combine(NodeMapping const &, NodeMapping const &);
  static bool nodes_are_disjoint(NodeMapping const &m1, NodeMapping const &m2);

  req<std::unordered_map<Node, T>> mapping;
};

using MachineMapping = NodeMapping<MachineView>;
using MemoryDecision = NodeMapping<OpMemoryDecision>;

FF_VISITABLE_STRUCT(MachineMapping, mapping);
FF_VISITABLE_STRUCT(MemoryDecision, mapping);

struct OptimalCostState {
  SerialParallelDecomposition subgraph;
  MachineSpecification resource;
  std::unordered_map<Node, MachineView> given_machine_views;
  req<std::unordered_map<OpenMultiDiEdge, MachineView>> frontier_machine_views;
};
FF_VISITABLE_STRUCT(OptimalCostState,
                    subgraph,
                    resource,
                    given_machine_views,
                    frontier_machine_views);

struct OptimalCostResult {
  static OptimalCostResult sequential_combine(OptimalCostResult const &s1,
                                              OptimalCostResult const &s2);
  static OptimalCostResult parallel_combine(OptimalCostResult const &s1,
                                            OptimalCostResult const &s2);
  static OptimalCostResult infinity();

  float runtime;
  req<MachineMapping> machine_mapping;
};
FF_VISITABLE_STRUCT(OptimalCostResult, runtime, machine_mapping);

struct OptimalCostRuntimeCmp {
  bool operator()(OptimalCostResult const &, OptimalCostResult const &);
};

template <typename T>
class OptimalCostCache {
public:
  OptimalCostCache() = default;

  std::optional<T> load(OptimalCostState const &) const;
  void save(OptimalCostState const &, T const &);

private:
  std::unordered_map<OptimalCostState, T> cache;
};

template <typename T>
T optimal_cost(ParallelComputationGraph const &g,
                 std::function<std::unordered_set<MachineView>(
                     Operator const &, MachineSpecification const &)> const
                     &allowed_machine_views,
                 CostEstimator const &cost_estimator,
                 MachineSpecification const &resources,
                 OptimalCostCache<T> &cached_subgraph_costs);

struct MemoryConfig {
  static MemoryConfig combine(MemoryConfig const &, MemoryConfig const &);

  float residual_memory;
  req<float> temporary_memory;
};
FF_VISITABLE_STRUCT(MemoryConfig, residual_memory, temporary_memory);

struct MemoryResult {
  static MemoryResult sequential_combine(MemoryResult const &,
                                         MemoryResult const &);
  static MemoryResult parallel_combine(MemoryResult const &,
                                       MemoryResult const &);

  float runtime;
  MachineMapping machine_mapping;
  MemoryDecision memory_decision;
};
FF_VISITABLE_STRUCT(MemoryResult, runtime, machine_mapping, memory_decision);

struct OptimalCostResultWithMemory {
  static OptimalCostResultWithMemory
      sequential_combine(OptimalCostResultWithMemory const &s1,
                         OptimalCostResultWithMemory const &s2);
  static OptimalCostResultWithMemory
      parallel_combine(OptimalCostResultWithMemory const &s1,
                       OptimalCostResultWithMemory const &s2);
  static OptimalCostResultWithMemory infinity();

  req<std::unordered_map<MemoryConfig, MemoryResult>> results;
};
FF_VISITABLE_STRUCT(OptimalCostResultWithMemory, results);

} // namespace FlexFlow

namespace std {

template <>
struct hash<std::unordered_map<FlexFlow::Node, FlexFlow::MachineMapping>> {
  size_t operator()(
      std::unordered_map<FlexFlow::Node, FlexFlow::MachineMapping> const &g)
      const;
};

}; // namespace std

#endif
