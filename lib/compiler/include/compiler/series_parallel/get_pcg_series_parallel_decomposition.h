#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_SERIES_PARALLEL_GET_PCG_SERIES_PARALLEL_DECOMPOSITION_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_SERIES_PARALLEL_GET_PCG_SERIES_PARALLEL_DECOMPOSITION_H

#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.dtg.h"

namespace FlexFlow {

std::optional<SeriesParallelDecomposition>
    get_pcg_series_parallel_decomposition(ParallelComputationGraph const &);

} // namespace FlexFlow

#endif
