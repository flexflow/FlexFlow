#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_FLIPPED_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_FLIPPED_H

#include "utils/graph/digraph/digraph_view.h"

namespace FlexFlow {

struct FlippedView : public IDiGraphView {
public:
  FlippedView() = delete;
  explicit FlippedView(DiGraphView const &);

  std::unordered_set<DirectedEdge>
      query_edges(DirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

  FlippedView *clone() const override;

private:
  DiGraphView g;
};

DiGraphView flipped(DiGraphView const &); 
DirectedEdge flipped_directed_edge(DirectedEdge const &);

} // namespace FlexFlow

#endif
