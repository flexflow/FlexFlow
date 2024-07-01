#ifndef _FLEXFLOW_UTILS_GRAPH_SERIALPARALLEL_H
#define _FLEXFLOW_UTILS_GRAPH_SERIALPARALLEL_H

#include "digraph.h"
#include "multidigraph.h"
#include "utils/optional.h"
#include <variant>
#include <vector>

namespace FlexFlow {
bool has_single_source(DiGraphView const &g);
bool has_single_sink(DiGraphView const &g);

Node find_source_node(DiGraphView const &);
Node find_sink_node(DiGraphView const &);

std::optional<Node> find_bottleneck_node(DiGraphView const &);

struct Parallel;

struct Serial {
  std::vector<std::variant<Parallel, Node>> children;
};

struct Parallel {
  std::vector<std::variant<Serial, Node>> children;
};

FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(Parallel, children);
FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(Serial, children);

using SerialParallelDecomposition = std::variant<Serial, Parallel, Node>;

SerialParallelDecomposition
    get_serial_parallel_decomposition(DiGraphView const &);

std::unordered_set<Node> get_nodes(SerialParallelDecomposition const &sp);

std::unordered_map<Node, Node> parallel_extend(MultiDiGraph &g,
                                               MultiDiGraph const &ext);

std::unordered_map<Node, Node> serial_extend(MultiDiGraph &g,
                                             MultiDiGraph const &ext);

MultiDiGraph serial_composition(MultiDiGraph const &g1, MultiDiGraph const &g2);

MultiDiGraph parallel_composition(MultiDiGraph const &g1,
                                  MultiDiGraph const &g2);

SerialParallelDecomposition parallel_composition(const std::vector<SerialParallelDecomposition>& sp_compositions);
SerialParallelDecomposition serial_composition(const std::vector<SerialParallelDecomposition>& sp_compositions);


MultiDiGraph multidigraph_from_sp_decomposition(
    SerialParallelDecomposition const &sp_decomposition);

} // namespace FlexFlow

#endif
