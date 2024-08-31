#include "utils/graph/open_dataflow_graph/algorithms/open_dataflow_graphs_are_isomorphic.h"
#include "utils/graph/open_dataflow_graph/algorithms/find_isomorphism.h"

namespace FlexFlow {

bool open_dataflow_graphs_are_isomorphic(OpenDataflowGraphView const &src,
                                         OpenDataflowGraphView const &dst) {
  return find_isomorphism(src, dst).has_value();
}

} // namespace FlexFlow
