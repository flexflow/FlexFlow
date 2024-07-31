#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_TRANSITIVE_REDUCTION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_TRANSITIVE_REDUCTION_H

#include "utils/graph/digraph/digraph_view.h"

namespace FlexFlow {

struct DirectedEdgeMaskView final : public IDiGraphView {
  DirectedEdgeMaskView() = delete;
  explicit DirectedEdgeMaskView(DiGraphView const &,
                                std::unordered_set<DirectedEdge> const &);

  std::unordered_set<DirectedEdge>
      query_edges(DirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

  DirectedEdgeMaskView *clone() const override;

private:
  DiGraphView g;
  std::unordered_set<DirectedEdge> edge_mask;
};

DiGraphView transitive_reduction(DiGraphView const &);

} // namespace FlexFlow

#endif
