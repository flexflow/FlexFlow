#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_GRAPH_GENERATION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_GRAPH_GENERATION_H

#include "utils/graph/dataflow_graph/dataflow_graph.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.dtg.h"

namespace FlexFlow {

void parallel_extend_unsafe(DataflowGraph &g, DataflowGraphView const &ext);

void series_extend(DataflowGraph &g, DataflowGraphView const &ext);

DataflowGraph series_composition(DataflowGraphView const &g1,
                                 DataflowGraphView const &g2);

DataflowGraph parallel_composition(DataflowGraphView const &g1,
                                   DataflowGraphView const &g2);

DataflowGraph dataflow_graph_from_sp_decomposition(
    SeriesParallelDecomposition const &sp_decomposition);

} // namespace FlexFlow

#endif
