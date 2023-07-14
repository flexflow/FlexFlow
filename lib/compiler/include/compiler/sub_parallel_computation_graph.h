#ifndef _FLEXFLOW_COMPILER_SUB_PARALLEL_COMPUTATION_GRAPH_H
#define _FLEXFLOW_COMPILER_SUB_PARALLEL_COMPUTATION_GRAPH_H

#include "op-attrs/operator_attrs.h"
#include "pcg/machine_view.h"
#include "pcg/operator.h"
#include "pcg/parallel_tensor.h"
#include "utils/graph.h"

namespace FlexFlow {

struct SubParallelComputationGraph
    : strong_typedef<
          SubParallelComputationGraph,
          LabelledOpenMultiDiGraph<Operator, ParallelTensor, MachineView>> {
  using strong_typedef::strong_typedef;
};

CHECK_WELL_BEHAVED_VALUE_TYPE(SubParallelComputationGraph);

} // namespace FlexFlow

#endif /* _FLEXFLOW_COMPILER_SUB_PARALLEL_COMPUTATION_GRAPH_H */
