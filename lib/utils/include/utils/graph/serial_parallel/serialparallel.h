#ifndef _FLEXFLOW_UTILS_GRAPH_SERIALPARALLEL_H
#define _FLEXFLOW_UTILS_GRAPH_SERIALPARALLEL_H

#include "utils/graph/digraph/digraph.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"
#include "utils/optional.h"
#include <variant>
#include <vector>

namespace FlexFlow {

SerialParallelDecomposition
    get_serial_parallel_decomposition(DiGraphView const &);

std::unordered_set<Node> get_nodes(SerialParallelDecomposition const &sp);
std::unordered_set<Node> get_nodes(SerialSplit const &);
std::unordered_set<Node> get_nodes(ParallelSplit const &);
std::unordered_set<Node> get_nodes(Node const &);

} // namespace FlexFlow

#endif
