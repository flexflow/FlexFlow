#ifndef _FLEXFLOW_RUNTIME_SRC_PARALLEL_COMPUTATION_GRAPH_H
#define _FLEXFLOW_RUNTIME_SRC_PARALLEL_COMPUTATION_GRAPH_H

#include "operator.h"
#include "parallel_tensor.h"
#include "utils/graph.h"

namespace FlexFlow {

class ParallelComputationGraph
    : public strong_typedef<
          ParallelComputationGraph,
          OutputLabelledMultiDiGraph<Operator, ParallelTensor>> {
  using strong_typedef::strong_typedef;
};

} // namespace FlexFlow

MAKE_TYPEDEF_HASHABLE(::FlexFlow::ParallelComputationGraph);

namespace FlexFlow {
static_assert(is_well_behaved_value_type<ParallelComputationGraph>::value, "");
}

#endif
