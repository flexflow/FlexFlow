#include "utils/graph/dataflow_graph/algorithms/dataflow_graphs_are_isomorphic.h"
#include "utils/graph/dataflow_graph/algorithms/find_isomorphism.h"

namespace FlexFlow {

bool dataflow_graphs_are_isomorphic(DataflowGraphView const &src,
                                    DataflowGraphView const &dst) {
  return find_isomorphism(src, dst).has_value();
}

} // namespace FlexFlow
