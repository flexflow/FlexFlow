#ifndef _FLEXFLOW_FFC_UNITY_ALGORITHM_H
#define _FLEXFLOW_FFC_UNITY_ALGORITHM_H

#include "op-attrs/operator_attrs.h"
#include "pcg/machine_view.h"
#include "pcg/machine_specification.h"
#include "utils/graph.h"
#include "compiler.h"

namespace FlexFlow {

/* std::unordered_map<MultiDiEdge, ParallelTensorShape>
 * infer_tensor_shapes(ParallelComputationGraph const &); */

/* std::unordered_set<Node> get_nodes(Serial const &serial); */
/* std::unordered_set<Node> get_nodes(Parallel const &parallel); */
/* std::unordered_set<Node> get_nodes(Node const &node); */
/* std::unordered_set<Node> get_nodes(SerialParallelDecomposition const &sp); */

/* float optimal_cost(ParallelComputationGraph const &g,
 * std::unordered_set<MachineView> const &allowed_machine_views); */
/* float optimal_cost(ParallelComputationGraph const &g, */
/*                    SerialParallelDecomposition const &, */
/*                    std::unordered_set<MachineView> const
 * &allowed_machine_views); */

struct ICostEstimator {
  virtual float estimate_cost(PCGOperatorAttrs const &op,
                              std::vector<ParallelTensorShape> const &inputs,
                              MachineView const &mv) const = 0;
  virtual float estimate_cost(ParallelTensorShape const &tensor_shape,
                              MachineView const &src,
                              MachineView const &dst) const = 0;
};

std::unordered_set<Node> get_closed_sources(OpenMultiDiGraphView const &g);
std::unordered_set<Node> get_closed_sinks(OpenMultiDiGraphView const &g);
std::unordered_set<Node> get_open_sources(OpenMultiDiGraphView const &g);
std::unordered_set<Node> get_open_sinks(OpenMultiDiGraphView const &g);

std::unordered_set<MultiDiEdge> get_cut(OpenMultiDiGraphView const &g,
                                        GraphSplit const &split);

using SubParallelComputationGraph =
    LabelledOpenMultiDiGraph<PCGOperatorAttrs,
                             ParallelTensorShape,
                             MachineView>;

enum class InputSettings { INCLUDE_INPUTS, EXCLUDE_INPUTS };

enum class OutputSettings { INCLUDE_OUTPUTS, EXCLUDE_OUTPUTS };

template <typename NodeLabel,
          typename EdgeLabel,
          typename InputLabel = EdgeLabel,
          typename OutputLabel = EdgeLabel>
LabelledOpenMultiDiGraph<NodeLabel, EdgeLabel, InputLabel, OutputLabel>
    get_subgraph(LabelledOpenMultiDiGraph<NodeLabel,
                                          EdgeLabel,
                                          InputLabel,
                                          OutputLabel> const &g,
                 std::unordered_set<Node> const &nodes,
                 InputSettings input_settings,
                 OutputSettings output_settings);

struct Strategy {
  Strategy(float runtime, std::unordered_map<Node, MachineView> machine_views);
  bool operator<(Strategy const &s) const;

  static Strategy sequential_combine(Strategy const &s1, Strategy const &s2);
  static Strategy parallel_combine(Strategy const &s1, Strategy const &s2);
  static Strategy infinity();

  float runtime;
  std::unordered_map<Node, MachineView> machine_views;
};

Strategy
    optimal_cost(OptimizerPCG const &g,
                 std::function<std::unordered_set<MachineView>(
                     PCGOperatorAttrs const &, MachineSpecification const &)> const
                     &allowed_machine_views,
                 ICostEstimator const &cost_estimator,
                 MachineSpecification const &resources,
                 std::unordered_map<size_t, Strategy> &cached_subgraph_costs);

struct GraphOptResult {
  OptimizerPCG pcg;
  Strategy strategy;

  GraphOptResult(OptimizerPCG const &pcg, Strategy const &strategy);

  bool operator<(GraphOptResult const &r) const;
};

GraphOptResult
    graph_optimize(OptimizerComputationGraph &cg,
                   ICostEstimator const &cost_estimator,
                   MachineSpecification const &resources,
                   std::function<std::unordered_set<MachineView>(
                       PCGOperatorAttrs const &, MachineSpecification const &)> const
                       &allowed_machine_views,
                   OptimizerConfig const &opt_config);
                   
} // namespace FlexFlow

namespace std {
template <>
struct hash<::FlexFlow::Serial> {
  size_t operator()(::FlexFlow::Serial const &) const; 
};

template <>
struct hash<::FlexFlow::Parallel> {
  size_t operator()(::FlexFlow::Parallel const &) const;
};

template <>
struct hash<::FlexFlow::GraphOptResult> {
  size_t operator()(::FlexFlow::GraphOptResult const &) const;
}

}; // namespace std


#endif
