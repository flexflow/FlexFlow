#ifndef _FLEXFLOW_SUBSTITUTIONS_SUBSTITUTIONS_H
#define _FLEXFLOW_SUBSTITUTIONS_SUBSTITUTIONS_H

#include "graph_pattern_match.h"

namespace FlexFlow {

struct GraphPattern
    : public strong_typedef<
          GraphPattern,
          LabelledOpenMultiDiGraph<OperatorPattern, ParallelTensorPattern>> {
  using strong_typedef::strong_typedef;
};

GraphSplit split_pattern(OpenMultiDiGraphView const &pattern);

bool is_singleton_pattern(OpenMultiDiGraphView const &);

bool assignment_satisfies(ParallelComputationGraph const &,
                          GraphPattern const &,
                          DiGraphPatternMatch const &);

} // namespace FlexFlow

#endif
