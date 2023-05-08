#ifndef _FLEXFLOW_FFC_UNITY_ALGORITHM_H
#define _FLEXFLOW_FFC_UNITY_ALGORITHM_H

#include "utils/graph.h"
#include "op-attrs/operator_attrs.h"
#include "pcg/machine_view.h"

namespace FlexFlow {

/* std::unordered_map<MultiDiEdge, ParallelTensorShape> infer_tensor_shapes(ParallelComputationGraph const &); */

/* std::unordered_set<Node> get_nodes(Serial const &serial); */
/* std::unordered_set<Node> get_nodes(Parallel const &parallel); */
/* std::unordered_set<Node> get_nodes(Node const &node); */
/* std::unordered_set<Node> get_nodes(SerialParallelDecomposition const &sp); */

/* float optimal_cost(ParallelComputationGraph const &g, std::unordered_set<MachineView> const &allowed_machine_views); */
/* float optimal_cost(ParallelComputationGraph const &g, */ 
/*                    SerialParallelDecomposition const &, */ 
/*                    std::unordered_set<MachineView> const &allowed_machine_views); */

struct ICostEstimator {
  virtual float estimate_cost(PCGOperatorAttrs const &op, 
                              std::vector<ParallelTensorShape> const &inputs, 
                              MachineView const &mv) const = 0;
  virtual float estimate_cost(ParallelTensorShape const &tensor_shape,
                              MachineView const &src, 
                              MachineView const &dst) = 0;
};

std::unordered_set<Node> get_closed_sources(OpenMultiDiGraphView const &g);
std::unordered_set<Node> get_closed_sinks(OpenMultiDiGraphView const &g);
std::unordered_set<Node> get_open_sources(OpenMultiDiGraphView const &g);
std::unordered_set<Node> get_open_sinks(OpenMultiDiGraphView const &g);

std::unordered_set<MultiDiEdge> get_cut(OpenMultiDiGraphView const &g, GraphSplit const &split);

using SubParallelComputationGraph = LabelledOpenMultiDiGraph<PCGOperatorAttrs, ParallelTensorShape, MachineView>;

SubParallelComputationGraph get_subgraph(SubParallelComputationGraph const &g,
                                        std::unordered_set<Node> const &nodes,
                                        InputSettings input_settings,
                                        OutputSettings output_settings);

enum class InputSettings {
  INCLUDE_INPUTS,
  EXCLUDE_INPUTS
};

enum class OutputSettings {
  INCLUDE_OUTPUTS,
  EXCLUDE_OUTPUTS
};

struct ParallelComputationGraph {
  MultiDiGraphView const &graph() const;
  PCGOperatorAttrs const &at(Node const &) const;
};

struct Strategy {
  Strategy(float runtime, std::unordered_map<Node, MachineView> machine_views);
  bool operator<(Strategy const &s) const;

  static Strategy sequential_combine(Strategy const &s1, Strategy const &s2);
  static Strategy parallel_combine(Strategy const &s1, Strategy const &s2);
  static Strategy infinity();

  float runtime;
  std::unordered_map<Node, MachineView> machine_views;
};

Strategy optimal_cost(ParallelComputationGraph const &g,
                   SerialParallelDecomposition const &sp_decomposition,
                   std::function<std::unordered_set<MachineView>(PCGOperatorAttrs const &, MachineResource const &)> const &allowed_machine_views,
                   ICostEstimator const &cost_estimator,
                   MachineResource const &resources);

}

#endif 
