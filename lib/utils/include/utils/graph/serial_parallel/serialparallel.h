#ifndef _FLEXFLOW_UTILS_GRAPH_SERIALPARALLEL_H
#define _FLEXFLOW_UTILS_GRAPH_SERIALPARALLEL_H

#include "utils/graph/digraph/digraph.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"
#include "utils/optional.h"
#include <variant>
#include <vector>

namespace FlexFlow {

Node find_source_node(DiGraphView const &);
Node find_sink_node(DiGraphView const &);

std::optional<Node> find_bottleneck_node(DiGraphView const &);

struct Parallel;

SerialParallelDecomposition
    get_serial_parallel_decomposition(DiGraphView const &);

std::unordered_set<Node> get_nodes(SerialParallelDecomposition const &sp);
std::unordered_set<Node> get_nodes(Serial const &);
std::unordered_set<Node> get_nodes(Parallel const &);
std::unordered_set<Node> get_nodes(Node const &);

// std::unordered_map<Node, Node> parallel_extend(MultiDiGraph &g,
//                                                MultiDiGraph const &ext);

// std::unordered_map<Node, Node> serial_extend(MultiDiGraph &g,
//                                              MultiDiGraph const &ext);

// MultiDiGraph serial_composition(MultiDiGraph const &g1, MultiDiGraph const &g2);

// MultiDiGraph parallel_composition(MultiDiGraph const &g1,
//                                   MultiDiGraph const &g2);

// MultiDiGraph multidigraph_from_sp_decomposition(
//     SerialParallelDecomposition const &sp_decomposition);

} // namespace FlexFlow

#endif
