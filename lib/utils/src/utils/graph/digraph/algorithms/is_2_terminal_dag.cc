#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/digraph/algorithms/is_acyclic.h"

namespace FlexFlow {

bool is_2_terminal_dag(DiGraphView const &g) {
  return (is_acyclic(g) && (get_sources(g).size() == 1) &&
          get_sinks(g).size() == 1);
}

} // namespace FlexFlow
