#ifndef _FLEXFLOW_UTILS_GRAPH_SERIAL_PARALLEL_CRITICAL_PATH_PRESERVING_SP_IZATION_H
#define _FLEXFLOW_UTILS_GRAPH_SERIAL_PARALLEL_CRITICAL_PATH_PRESERVING_SP_IZATION_H

#include "utils/graph/digraph/digraph_view.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"
#include <unordered_map>

namespace FlexFlow {

SerialParallelDecomposition
    critical_path_preserving_sp_ization(DiGraphView const &g);

SerialParallelDecomposition
    critical_path_preserving_sp_ization_with_coalescing(DiGraphView const &g);

} // namespace FlexFlow

#endif
