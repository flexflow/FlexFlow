#include "utils/graph/dataflow_graph/algorithms/as_dot.h"
#include "utils/dot_file.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/node/algorithms.h"
#include "utils/record_formatter.h"

namespace FlexFlow {

// WARN(@lockshaw): doing this all with string ids is ugly and error prone,
// as it requires duplicating the stringification logic across functions.
//
// Fixing this is tracked in issue
std::string as_dot(DataflowGraphView const &g) {
  std::ostringstream oss;
  DotFile<std::string> dot = DotFile<std::string>{oss};

  std::function<std::string(Node const &)> get_node_label =
      [](Node const &n) -> std::string {
    return fmt::format("n{}", n.raw_uid);
  };
  as_dot(dot, g, get_node_label);

  dot.close();
  return oss.str();
}

void as_dot(DotFile<std::string> &dot,
            DataflowGraphView const &g,
            std::function<std::string(Node const &)> const &get_node_label) {
  auto get_node_name = [](Node n) { return fmt::format("n{}", n.raw_uid); };

  auto get_input_field = [](int idx) { return fmt::format("i{}", idx); };

  auto get_output_field = [](int idx) { return fmt::format("o{}", idx); };

  for (Node const &n : get_nodes(g)) {
    std::vector<DataflowInput> n_inputs = get_dataflow_inputs(g, n);
    std::vector<DataflowOutput> n_outputs = get_outputs(g, n);

    RecordFormatter inputs_record;
    for (DataflowInput const &i : n_inputs) {
      inputs_record << fmt::format("<{}>{}", get_input_field(i.idx), i.idx);
    }

    RecordFormatter outputs_record;
    for (DataflowOutput const &o : n_outputs) {
      outputs_record << fmt::format("<{}>{}", get_output_field(o.idx), o.idx);
    }

    RecordFormatter rec;
    rec << inputs_record << get_node_label(n) << outputs_record;

    dot.add_record_node(get_node_name(n), rec);
  }

  for (DataflowEdge const &e : get_edges(g)) {
    dot.add_edge(get_node_name(e.src.node),
                 get_node_name(e.dst.node),
                 get_output_field(e.src.idx),
                 get_input_field(e.dst.idx));
  }
}

} // namespace FlexFlow
