#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_FIND_ISOMORPHISM_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_FIND_ISOMORPHISM_H

#include "utils/graph/open_dataflow_graph/algorithms/open_dataflow_graph_isomorphism.dtg.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_graph_view.h"

namespace FlexFlow {

/**
 * @brief Find a valid isomorphism between \p src and \p dst, if one exists
 *
 * @note If multiple isomorphisms exist, an arbitrary one is returned
 */
std::optional<OpenDataflowGraphIsomorphism>
    find_isomorphism(OpenDataflowGraphView const &src,
                     OpenDataflowGraphView const &dst);

} // namespace FlexFlow

#endif
