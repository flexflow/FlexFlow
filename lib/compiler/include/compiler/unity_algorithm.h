#ifndef _FLEXFLOW_FFC_UNITY_ALGORITHM_H
#define _FLEXFLOW_FFC_UNITY_ALGORITHM_H

#include "op-meta/operator_attrs.h"
#include "pcg/machine_view.h"
#include "utils/graph.h"

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

} // namespace FlexFlow

#endif
