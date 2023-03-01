#ifndef _FLEXFLOW_UTILS_GRAPH_SERIALPARALLEL_H
#define _FLEXFLOW_UTILS_GRAPH_SERIALPARALLEL_H

#include "digraph.h"
#include "multidigraph.h"
#include "mpark/variant.hpp"
#include <vector>

namespace FlexFlow {
namespace utils {

Node find_source_node(IDiGraphView const &);
Node find_sink_node(IDiGraphView const &);

tl::optional<Node> find_bottleneck_node(IMultiDiGraphView const &);
tl::optional<Node> find_bottleneck_node(IDiGraphView const &);

struct Parallel;

struct Serial {
  std::vector<mpark::variant<Parallel, Node>> children;
};

struct Parallel {
  std::vector<mpark::variant<Serial, Node>> children;
};

using SerialParallelDecomposition = mpark::variant<Serial, Parallel, Node>;

SerialParallelDecomposition get_serial_parallel_decomposition(IDiGraphView const &);

}
}

#endif 
