#ifndef _FLEXFLOW_COMPILER_MCMC_STATE_H
#define _FLEXFLOW_COMPILER_MCMC_STATE_H

#include "compiler/graph_optimize_result.dtg.h"

namespace FlexFlow {

struct GraphOptimizeState {
  GraphOptimizeState(GraphOptimizeResult const &graph_optimize_result,
                     float runtime);

  GraphOptimizeResult graph_optimize_result;
  float runtime;

  bool operator==(GraphOptimizeState const &other) const;
  bool operator!=(GraphOptimizeState const &other) const;
  bool operator<(GraphOptimizeState const &other) const;
};

} // namespace FlexFlow

namespace std {

template <>
struct hash<::FlexFlow::GraphOptimizeState> {
  size_t operator()(::FlexFlow::GraphOptimizeState const &) const;
};

} // namespace std

#endif
