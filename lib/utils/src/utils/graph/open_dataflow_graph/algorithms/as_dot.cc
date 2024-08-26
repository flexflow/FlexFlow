#include "utils/graph/open_dataflow_graph/algorithms/as_dot.h"
#include "utils/dot_file.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/dataflow_graph/algorithms/as_dot.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_incoming_edges.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_inputs.h"

namespace FlexFlow {

std::string as_dot(OpenDataflowGraphView const &g) {
  std::function<std::string(Node const &)> get_node_label = [](Node const &n) {
    return fmt::format("n{}", n.raw_uid);
  };

  std::function<std::string(DataflowGraphInput const &)> get_input_label =
      [](DataflowGraphInput const &i) { return fmt::format("i{}", i.idx); };

  return as_dot(g, get_node_label, get_input_label);
}

// WARN(@lockshaw): doing this all with string ids is ugly and error prone,
// as it requires duplicating the stringification logic across functions.
//
// Fixing this is tracked in issue https://github.com/flexflow/FlexFlow/issues/1476
std::string
    as_dot(OpenDataflowGraphView const &g,
           std::function<std::string(Node const &)> const &get_node_label,
           std::function<std::string(DataflowGraphInput const &)> const
               &get_input_label) {
  std::ostringstream oss;
  DotFile<std::string> dot = DotFile<std::string>{oss};

  as_dot(dot, static_cast<DataflowGraphView>(g), get_node_label);

  auto get_node_name = [](Node n) { return fmt::format("n{}", n.raw_uid); };

  auto get_input_field = [](int idx) { return fmt::format("i{}", idx); };

  auto get_output_field = [](int idx) { return fmt::format("o{}", idx); };

  auto get_graph_input_name = [](DataflowGraphInput i) {
    return fmt::format("gi{}", i.idx);
  };

  for (DataflowGraphInput const &i : get_inputs(g)) {
    dot.add_node(get_graph_input_name(i),
                 {{"style", "dashed"}, {"label", get_input_label(i)}});
  }

  for (DataflowInputEdge const &e : get_incoming_edges(g)) {
    dot.add_edge(get_graph_input_name(e.src),
                 get_node_name(e.dst.node),
                 std::nullopt,
                 get_input_field(e.dst.idx));
  }

  dot.close();
  return oss.str();
}

} // namespace FlexFlow
