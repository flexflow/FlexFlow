#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_INVERSE_LINE_GRAPH_GET_INVERSE_LINE_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_INVERSE_LINE_GRAPH_GET_INVERSE_LINE_GRAPH_H

#include "utils/graph/digraph/algorithms/inverse_line_graph/inverse_line_graph_result.dtg.h"
#include "utils/graph/digraph/digraph_view.h"

namespace FlexFlow {

std::optional<InverseLineGraphResult>
    get_inverse_line_graph(DiGraphView const &);

} // namespace FlexFlow

#endif
