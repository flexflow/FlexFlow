#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_FILE_FORMAT_V1_V1_PARALLEL_COMPUTATION_GRAPH_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_FILE_FORMAT_V1_V1_PARALLEL_COMPUTATION_GRAPH_H

#include "pcg/file_format/v1/v1_parallel_computation_graph.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"

namespace FlexFlow {

V1ParallelComputationGraph to_v1(ParallelComputationGraph const &);

} // namespace FlexFlow

#endif
