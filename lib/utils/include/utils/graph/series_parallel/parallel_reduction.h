#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_PARALLEL_REDUCTION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_PARALLEL_REDUCTION_H

#include "utils/graph/multidigraph/multidigraph.h"
#include "utils/graph/series_parallel/parallel_reduction.dtg.h"
#include <optional>

namespace FlexFlow {

ParallelReduction make_parallel_reduction(MultiDiEdge const &,
                                          MultiDiEdge const &);
std::optional<ParallelReduction>
    find_parallel_reduction(MultiDiGraphView const &);

MultiDiEdge apply_parallel_reduction(MultiDiGraph &, ParallelReduction const &);

} // namespace FlexFlow

#endif
