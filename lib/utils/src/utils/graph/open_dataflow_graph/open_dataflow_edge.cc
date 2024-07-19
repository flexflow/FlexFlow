#include "utils/graph/open_dataflow_graph/open_dataflow_edge.h"
#include "utils/overload.h"

namespace FlexFlow {

Node get_open_dataflow_edge_dst_node(OpenDataflowEdge const &e) {
  return e.visit<Node>(overload{
      [](DataflowEdge const &e) { return e.dst.node; },
      [](DataflowInputEdge const &e) { return e.dst.node; },
  });
}

int get_open_dataflow_edge_dst_idx(OpenDataflowEdge const &e) {
  return e.visit<int>(overload{
      [](DataflowEdge const &e) { return e.dst.idx; },
      [](DataflowInputEdge const &e) { return e.dst.idx; },
  });
}

OpenDataflowValue
    get_open_dataflow_edge_source(OpenDataflowEdge const &open_e) {
  return open_e.visit<OpenDataflowValue>(overload{
      [](DataflowEdge const &e) { return OpenDataflowValue{e.src}; },
      [](DataflowInputEdge const &e) { return OpenDataflowValue{e.src}; },
  });
}

OpenDataflowEdge
    open_dataflow_edge_from_src_and_dst(OpenDataflowValue const &src,
                                        DataflowInput const &dst) {
  return src.visit<OpenDataflowEdge>(overload{
      [&](DataflowOutput const &o) {
        return OpenDataflowEdge{DataflowEdge{o, dst}};
      },
      [&](DataflowGraphInput const &gi) {
        return OpenDataflowEdge{DataflowInputEdge{gi, dst}};
      },
  });
}

} // namespace FlexFlow
