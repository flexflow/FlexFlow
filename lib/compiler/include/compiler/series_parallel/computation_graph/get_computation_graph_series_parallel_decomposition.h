#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_SERIES_PARALLEL_GET_COMPUTATION_GRAPH_SERIES_PARALLEL_DECOMPOSITION_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_SERIES_PARALLEL_GET_COMPUTATION_GRAPH_SERIES_PARALLEL_DECOMPOSITION_H

#include "pcg/computation_graph.dtg.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.dtg.h"

namespace FlexFlow {

std::string render_preprocessed_computation_graph_for_sp_decomposition(
    ComputationGraph const &);
std::optional<SeriesParallelDecomposition>
    get_computation_graph_series_parallel_decomposition(
        ComputationGraph const &);

} // namespace FlexFlow

#endif
