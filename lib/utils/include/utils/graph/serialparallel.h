#ifndef _FLEXFLOW_UTILS_GRAPH_SERIALPARALLEL_H
#define _FLEXFLOW_UTILS_GRAPH_SERIALPARALLEL_H

#include "digraph.h"
#include "multidigraph.h"
#include "utils/optional.h"
#include "utils/variant.h"
#include <vector>

namespace FlexFlow {

Node find_source_node(DiGraphView const &);
Node find_sink_node(DiGraphView const &);

optional<Node> find_bottleneck_node(MultiDiGraphView const &);
optional<Node> find_bottleneck_node(DiGraphView const &);

struct Parallel;

struct Serial {
  std::vector<variant<Parallel, Node>> children;
};

struct Parallel {
  std::vector<variant<Serial, Node>> children;
};

FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(Parallel, children);
FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(Serial, children);

using SerialParallelDecomposition = variant<Serial, Parallel, Node>;

SerialParallelDecomposition
    get_serial_parallel_decomposition(DiGraphView const &);

std::unordered_set<Node> get_nodes(SerialParallelDecomposition const &sp);

} // namespace FlexFlow

#endif
