#ifndef _FLEXFLOW_UTILS_GRAPH_barrier_sync_sp_ization_H
#define _FLEXFLOW_UTILS_GRAPH_barrier_sync_sp_ization_H

#include "serialparallel.h"

using namespace FlexFlow;

namespace FlexFlow {

SerialParallelDecomposition barrier_sync_sp_ization(DiGraphView const &g);
SerialParallelDecomposition
    naive_dependency_invariant_sp_ization(DiGraphView const &g);
} // namespace FlexFlow

#endif
