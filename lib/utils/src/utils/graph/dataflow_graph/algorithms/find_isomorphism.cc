#include "utils/graph/dataflow_graph/algorithms/find_isomorphism.h"
#include "utils/containers/get_first.h"
#include "utils/graph/dataflow_graph/algorithms/find_isomorphisms.h"

namespace FlexFlow {

std::optional<DataflowGraphIsomorphism>
    find_isomorphism(DataflowGraphView const &src,
                     DataflowGraphView const &dst) {
  std::unordered_set<DataflowGraphIsomorphism> all_isomorphisms =
      find_isomorphisms(src, dst);

  if (all_isomorphisms.empty()) {
    return std::nullopt;
  } else {
    return get_first(all_isomorphisms);
  }
}

} // namespace FlexFlow
