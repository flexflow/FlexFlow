#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_PARALLEL_COMPUTATION_GRAPH_BUILDER_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_PARALLEL_COMPUTATION_GRAPH_BUILDER_H

#include "pcg/parallel_computation_graph.dtg.h"
#include "pcg/parallel_tensor_guid_t.dtg.h"

namespace FlexFlow {

struct ParallelComputationGraphBuilder {
public:
  ParallelComputationGraphBuilder();


public:
  ParallelComputationGraph pcg;
};

} // namespace FlexFlow

#endif
