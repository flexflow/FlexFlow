#include "utils/graph/digraph/algorithms/digraph_as_dot.h"
#include "utils/dot_file.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/digraph/algorithms.h"

namespace FlexFlow {

std::string digraph_as_dot(DiGraphView const &g,
                   std::function<std::string(Node const &)> const &get_node_label) {
  std::ostringstream oss;
  DotFile<std::string> dot = DotFile<std::string>{oss};
  
  auto get_node_name = [](Node const &n) { 
    return fmt::format("n{}", n.raw_uid);
  };

  for (Node const &n : get_nodes(g)) {
    RecordFormatter rec;
    rec << get_node_label(n);
    dot.add_record_node(get_node_name(n), rec);
  }

  for (DirectedEdge const &e : get_edges(g)) {
    dot.add_edge(get_node_name(e.src), get_node_name(e.dst));
  }

  dot.close();
  return oss.str();
}

} // namespace FlexFlow
