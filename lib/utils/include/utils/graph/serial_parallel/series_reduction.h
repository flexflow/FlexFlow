#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_SERIES_REDUCTION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_SERIES_REDUCTION_H

#include "utils/graph/multidigraph/multidiedge.dtg.h"
#include "utils/graph/serial_parallel/series_reduction.dtg.h"
#include "utils/graph/multidigraph/multidigraph_view.h"

namespace FlexFlow {

SeriesReduction make_series_reduction(MultiDiEdge const &, MultiDiEdge const &);
std::optional<SeriesReduction> find_series_reduction(MultiDiGraphView const &);

} // namespace FlexFlow

#endif
