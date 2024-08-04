#ifndef _FLEXFLOW_UTILS_GRAPH_SERIAL_PARALLEL_WORK_PRESERVING_SP_IZATION_H
#define _FLEXFLOW_UTILS_GRAPH_SERIAL_PARALLEL_WORK_PRESERVING_SP_IZATION_H

#include "utils/graph/digraph/digraph_view.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"
#include <unordered_map>

namespace FlexFlow {

SerialParallelDecomposition barrier_sync_sp_ization(DiGraphView const &g);

SerialParallelDecomposition cost_aware_barrier_sync_sp_ization(
    DiGraphView const &g, std::unordered_map<Node, float> const &cost_map);

} // namespace FlexFlow

#endif
