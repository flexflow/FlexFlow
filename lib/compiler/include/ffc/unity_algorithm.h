#ifndef _FLEXFLOW_FFC_UNITY_ALGORITHM_H
#define _FLEXFLOW_FFC_UNITY_ALGORITHM_H

#include "utils/graph.h"
#include "op-meta/operator_params.h"
#include "pcg/machine_view.h"

namespace FlexFlow {
namespace ffc {

struct UnitySearchConfig { };

using SearchAlgorithmConfig = mpark::variant<
  UnitySearchConfig
>;

void run_search_algorithm(SearchAlgorithmConfig const &);

struct ParallelComputationGraph {
  ParallelComputationGraph() = delete;
  ParallelComputationGraph(utils::AdjacencyMultiDiGraph const &, 
                           std::unordered_map<utils::Node, opmeta::OperatorParameters> const &);

  utils::AdjacencyMultiDiGraph g;
  std::unordered_map<utils::Node, opmeta::OperatorParameters> nodeMap;
};


std::unordered_map<utils::MultiDiEdge, opmeta::ParallelTensorShape> infer_tensor_shapes(ParallelComputationGraph const &);

std::unordered_set<utils::Node> get_nodes(utils::Serial const &serial);
std::unordered_set<utils::Node> get_nodes(utils::Parallel const &parallel);
std::unordered_set<utils::Node> get_nodes(utils::Node const &node);
std::unordered_set<utils::Node> get_nodes(utils::SerialParallelDecomposition const &sp);

float optimal_cost(ParallelComputationGraph const &g, std::unordered_set<MachineView> const &allowed_machine_views);
float optimal_cost(ParallelComputationGraph const &g, 
                   utils::SerialParallelDecomposition const &, 
                   std::unordered_set<MachineView> const &allowed_machine_views);

}
}

#endif 
