#ifndef _FLEXFLOW_SUBSTITUTIONS_SUB_PARALLEL_COMPUTATION_GRAPH_H
#define _FLEXFLOW_SUBSTITUTIONS_SUB_PARALLEL_COMPUTATION_GRAPH_H

#include "op-attrs/operator_attrs.h"
#include "pcg/machine_view.h"
#include "pcg/operator.h"
#include "pcg/parallel_tensor.h"
#include "utils/graph.h"

namespace FlexFlow {

using SubParallelComputationGraph =
    LabelledOpenMultiDiGraph<Operator, ParallelTensor>;

CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(SubParallelComputationGraph);

ParallelTensor at(SubParallelComputationGraph const &g,
                  OpenMultiDiEdge const &e);

} // namespace FlexFlow

#endif
