#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_PARALLEL_COMPUTATION_GRAPH_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_PARALLEL_COMPUTATION_GRAPH_H

#include "operator.h"
#include "parallel_tensor.h"
#include "utils/graph.h"

namespace FlexFlow {

struct ParallelComputationGraph
    : public strong_typedef<
          ParallelComputationGraph,
          OutputLabelledMultiDiGraph<Operator, ParallelTensor>> {
  using strong_typedef::strong_typedef;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_HASH(ParallelComputationGraph);

bool operator==(ParallelComputationGraph const &,
                ParallelComputationGraph const &);

} // namespace FlexFlow

namespace std {

template <>
struct hash<FlexFlow::ParallelComputationGraph> {
  size_t operator()(FlexFlow::ParallelComputationGraph const &g) const;
};
} // namespace std

#endif
