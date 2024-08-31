#include "utils/graph/dataflow_graph/algorithms/find_isomorphisms.h"
#include "utils/containers/transform.h"
#include "utils/graph/dataflow_graph/algorithms/view_as_open_dataflow_graph.h"
#include "utils/graph/open_dataflow_graph/algorithms/find_isomorphisms.h"

namespace FlexFlow {

std::unordered_set<DataflowGraphIsomorphism>
    find_isomorphisms(DataflowGraphView const &src,
                      DataflowGraphView const &dst) {
  std::unordered_set<OpenDataflowGraphIsomorphism> open_isomorphisms =
      find_isomorphisms(view_as_open_dataflow_graph(src),
                        view_as_open_dataflow_graph(dst));

  return transform(open_isomorphisms,
                   [](OpenDataflowGraphIsomorphism const &open) {
                     assert(open.input_mapping.empty());
                     return DataflowGraphIsomorphism{open.node_mapping};
                   });
}

} // namespace FlexFlow
