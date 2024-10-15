#include "utils/graph/open_dataflow_graph/algorithms/find_isomorphism.h"
#include "utils/containers/get_one_of.h"
#include "utils/graph/open_dataflow_graph/algorithms/find_isomorphisms.h"

namespace FlexFlow {

std::optional<OpenDataflowGraphIsomorphism>
    find_isomorphism(OpenDataflowGraphView const &src,
                     OpenDataflowGraphView const &dst) {
  std::unordered_set<OpenDataflowGraphIsomorphism> all_isomorphisms =
      find_isomorphisms(src, dst);

  if (all_isomorphisms.empty()) {
    return std::nullopt;
  } else {
    return get_one_of(all_isomorphisms);
  }
}

} // namespace FlexFlow
