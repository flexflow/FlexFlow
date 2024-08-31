#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_ALGORITHMS_FIND_ISOMORPHISMS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_ALGORITHMS_FIND_ISOMORPHISMS_H

#include "utils/graph/dataflow_graph/algorithms/dataflow_graph_isomorphism.dtg.h"
#include "utils/graph/dataflow_graph/dataflow_graph_view.h"

namespace FlexFlow {

std::unordered_set<DataflowGraphIsomorphism>
  find_isomorphisms(DataflowGraphView const &,
                    DataflowGraphView const &);

} // namespace FlexFlow

#endif
