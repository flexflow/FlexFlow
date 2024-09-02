#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_FILE_FORMAT_V1_V1_COMPUTATION_GRAPH_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_FILE_FORMAT_V1_V1_COMPUTATION_GRAPH_H

#include "pcg/file_format/v1/v1_computation_graph.dtg.h"
#include "pcg/computation_graph.dtg.h"

namespace FlexFlow {

V1ComputationGraph to_v1(ComputationGraph const &);

} // namespace FlexFlow

#endif
