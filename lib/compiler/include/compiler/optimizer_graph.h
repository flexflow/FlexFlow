#ifndef _FLEXFLOW_COMPILER_OPTIMIZER_GRAPH_H
#define _FLEXFLOW_COMPILER_OPTIMIZER_GRAPH_H

#include "op-attrs/operator_attrs.h"
#include "pcg/machine_view.h"
#include "utils/graph.h"

namespace FlexFlow {

using OptimizerComputationGraph =
    NodeLabelledMultiDiGraph<ComputationGraphAttrs>;
using OptimizerPCG =
    LabelledMultiDiGraph<PCGOperatorAttrs, ParallelTensorShape>;

using SubParallelComputationGraph =
    LabelledOpenMultiDiGraph<PCGOperatorAttrs,
                             ParallelTensorShape,
                             MachineView>;

}

#endif /* _FLEXFLOW_COMPILER_OPTIMIZER_GRAPH_H */
