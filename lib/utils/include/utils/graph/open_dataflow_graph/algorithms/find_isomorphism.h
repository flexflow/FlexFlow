#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_FIND_ISOMORPHISM_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_FIND_ISOMORPHISM_H

#include "utils/graph/open_dataflow_graph/algorithms/open_dataflow_graph_isomorphism.dtg.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_graph_view.h"

namespace FlexFlow {

std::unordered_set<OpenDataflowGraphIsomorphism>
    find_isomorphisms(OpenDataflowGraphView const &,
                      OpenDataflowGraphView const &);
std::optional<OpenDataflowGraphIsomorphism>
    find_isomorphism(OpenDataflowGraphView const &,
                     OpenDataflowGraphView const &);

} // namespace FlexFlow

#endif
