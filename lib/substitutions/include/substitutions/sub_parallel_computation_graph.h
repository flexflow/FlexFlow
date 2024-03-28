#ifndef _FLEXFLOW_SUBSTITUTIONS_SUB_PARALLEL_COMPUTATION_GRAPH_H
#define _FLEXFLOW_SUBSTITUTIONS_SUB_PARALLEL_COMPUTATION_GRAPH_H

#include "op-attrs/operator_attrs.h"
#include "pcg/machine_view.h"
#include "pcg/operator.h"
#include "pcg/parallel_tensor.h"
#include "utils/graph.h"

namespace FlexFlow {

/**
 * @brief SubParallelComputationGraph is defined as an open graph, which allows nodes and edges 
 * that are not from the same graph to be added to it.
 * This definition is useful when we want to split and merge graphs when doing pattern matching.
 * In contrast, the ParallelComputationGraph is defined as a closed graph and all the edges and 
 * nodes are within that graph.
 */
using SubParallelComputationGraph =
    OutputLabelledOpenMultiDiGraph<Operator, ParallelTensor>;

CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(SubParallelComputationGraph);

} // namespace FlexFlow

#endif
