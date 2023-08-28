#ifndef _FLEXFLOW_SUBSTITUTIONS_SUBSTITUTIONS_H
#define _FLEXFLOW_SUBSTITUTIONS_SUBSTITUTIONS_H

#include "graph_pattern_match.h"
#include "operator_pattern.h"
#include "parallel_tensor_pattern.h"
#include "pcg/parallel_computation_graph.h"

namespace FlexFlow {

struct GraphPattern
    : public strong_typedef<
          GraphPattern,
          OutputLabelledOpenMultiDiGraph<OperatorPattern,
                                         ParallelTensorPattern>> {
  using strong_typedef::strong_typedef;
};

GraphSplit split_pattern(OpenMultiDiGraphView const &pattern);

bool is_singleton_pattern(OpenMultiDiGraphView const &);

bool assignment_satisfies(ParallelComputationGraph const &,
                          GraphPattern const &,
                          MultiDiGraphPatternMatch const &);

} // namespace FlexFlow

#endif
