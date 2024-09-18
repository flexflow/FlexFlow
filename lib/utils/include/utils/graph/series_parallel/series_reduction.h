#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_SERIES_REDUCTION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_SERIES_REDUCTION_H

#include "utils/graph/multidigraph/multidiedge.dtg.h"
#include "utils/graph/multidigraph/multidigraph.h"
#include "utils/graph/series_parallel/series_reduction.dtg.h"

namespace FlexFlow {

Node get_pre_node(MultiDiGraphView const &, SeriesReduction const &);
Node get_post_node(MultiDiGraphView const &, SeriesReduction const &);
Node get_center_node(MultiDiGraphView const &, SeriesReduction const &);

SeriesReduction make_series_reduction(MultiDiEdge const &, MultiDiEdge const &);
std::optional<SeriesReduction> find_series_reduction(MultiDiGraphView const &);

MultiDiEdge apply_series_reduction(MultiDiGraph &, SeriesReduction const &);

} // namespace FlexFlow

#endif
