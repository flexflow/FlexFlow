#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_ALGORITHMS_FIND_ISOMORPHISM_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_ALGORITHMS_FIND_ISOMORPHISM_H

#include "utils/containers/zip.h"
#include "utils/containers/as_vector.h"
#include "utils/containers/get_all_permutations.h"
#include "utils/graph/labelled_open_dataflow_graph/labelled_open_dataflow_graph_view.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/is_isomorphic_under.h"

namespace FlexFlow {

template <typename NodeLabel, typename EdgeLabel>
std::optional<bidict<Node, Node>> find_isomorphism(LabelledOpenDataflowGraphView<NodeLabel, EdgeLabel> const &src,
                                    LabelledOpenDataflowGraphView<NodeLabel, EdgeLabel> const &dst) {
  std::vector<Node> src_nodes = as_vector(get_nodes(src));
  for (std::vector<Node> const &dst_nodes : get_all_permutations(get_nodes(dst))) {
    std::vector<std::pair<Node, Node>> zipped = zip(src_nodes, dst_nodes);
    bidict<Node, Node> candidate_isomorphism{zipped.cbegin(), zipped.cend()};

    if (is_isomorphic_under(src, dst, candidate_isomorphism)) {
      return candidate_isomorphism;
    }
  }
  
  return std::nullopt;
}

} // namespace FlexFlow

#endif
