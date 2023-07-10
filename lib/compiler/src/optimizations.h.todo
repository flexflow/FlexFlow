#ifndef _FLEXFLOW_RUNTIME_SRC_OPTIMIZATIONS_H
#define _FLEXFLOW_RUNTIME_SRC_OPTIMIZATIONS_H

#include "parallel_computation_graph.h"

namespace FlexFlow {

ParallelComputationGraph fuse_operators(ParallelComputationGraph const &);
ParallelComputationGraph
    remove_unnecessary_gradient_calculations(ParallelComputationGraph const &);
ParallelComputationGraph
    enable_inplace_operators(ParallelComputationGraph const &);

} // namespace FlexFlow

#endif
