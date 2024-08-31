#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_ALGORITHMS_FIND_ISOMORPHISM_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_ALGORITHMS_FIND_ISOMORPHISM_H

#include "utils/graph/dataflow_graph/algorithms/dataflow_graph_isomorphism.dtg.h"
#include "utils/graph/dataflow_graph/dataflow_graph_view.h"
#include <optional>

namespace FlexFlow {

/**
 * @brief Find a valid isomorphism between \p src and \p dst, if one exists
 *
 * @note If multiple isomorphisms exist, an arbitrary one is returned
 */
std::optional<DataflowGraphIsomorphism> find_isomorphism(DataflowGraphView const &,
                                                         DataflowGraphView const &);

} // namespace FlexFlow

#endif
