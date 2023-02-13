#ifndef _FLEXFLOW_UTILS_GRAPH_VIEWS_H
#define _FLEXFLOW_UTILS_GRAPH_VIEWS_H

#include "digraph.h"

namespace FlexFlow {
namespace utils {

struct FlippedView : public IDiGraphView {
public:
  FlippedView() = delete;
  explicit FlippedView(IDiGraphView const &);

  std::unordered_set<DirectedEdge> query_edges(DirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;
private:
  IDiGraphView const *g;
};

DirectedEdge flipped(DirectedEdge const &);

FlippedView unsafe_view_as_flipped(IDiGraphView const &);

}
}

#endif 
