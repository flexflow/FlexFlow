#include "utils/graph/open_dataflow_graph/open_dataflow_edge.h"
#include "utils/overload.h"

namespace FlexFlow {

int get_open_dataflow_edge_dst_idx(OpenDataflowEdge const &e) {
  return e.visit<int>(overload {
    [](DataflowEdge const &e) { return e.dst.idx; },
    [](DataflowInputEdge const &e) { return e.dst.idx; },
  });
}

} // namespace FlexFlow
