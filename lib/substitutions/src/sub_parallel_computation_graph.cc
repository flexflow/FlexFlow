#include "substitutions/sub_parallel_computation_graph.h"

namespace FlexFlow {

ParallelTensor at(SubParallelComputationGraph const &g,
                  OpenMultiDiEdge const &e) {
  return visit([&](auto const &e) { return g.at(e); }, e);
}

} // namespace FlexFlow
