#ifndef _FLEXFLOW_UTILS_GRAPH_SERIALPARALLEL_H
#define _FLEXFLOW_UTILS_GRAPH_SERIALPARALLEL_H

#include "digraph.h"
#include "multidigraph.h"
#include "utils/variant.h"
#include <vector>

namespace FlexFlow {

Node find_source_node(IDiGraphView const &);
Node find_sink_node(IDiGraphView const &);

optional<Node> find_bottleneck_node(IMultiDiGraphView const &);
optional<Node> find_bottleneck_node(IDiGraphView const &);

struct Parallel;

struct Serial {
  std::vector<variant<Parallel, Node>> children;
};

struct Parallel {
  std::vector<variant<Serial, Node>> children;
};

using SerialParallelDecomposition = variant<Serial, Parallel, Node>;

SerialParallelDecomposition get_serial_parallel_decomposition(IDiGraphView const &);

}

#endif 
