#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_ALGORITHMS_ARE_ISOMORPHIC_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_ALGORITHMS_ARE_ISOMORPHIC_H

#include "utils/graph/dataflow_graph/dataflow_graph_view.h"

namespace FlexFlow {

bool dataflow_graphs_are_isomorphic(DataflowGraphView const &,
                                    DataflowGraphView const &);

} // namespace FlexFlow

#endif
