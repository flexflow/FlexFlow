#ifndef _FLEXFLOW_UTILS_GRAPH_barrier_sync_sp_ization_H
#define _FLEXFLOW_UTILS_GRAPH_barrier_sync_sp_ization_H

#include "serialparallel.h"

using namespace FlexFlow;

namespace FlexFlow {

SerialParallelDecomposition barrier_sync_sp_ization(DiGraphView const &g);
SerialParallelDecomposition
    dependency_invariant_sp_ization(DiGraphView const &g);
SerialParallelDecomposition
    dependency_invariant_sp_ization_with_coalescing(DiGraphView const &g);

SerialParallelDecomposition cost_aware_barrier_sync_sp_ization(
    DiGraphView const &g, std::unordered_map<Node, float> const &cost_map);

} // namespace FlexFlow

#endif
