#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_CONTRACT_NODE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_CONTRACT_NODE_H

#include "utils/graph/digraph/digraph_view.h"

namespace FlexFlow {

struct ContractNodeView : public IDiGraphView {
  ContractNodeView() = delete;
  explicit ContractNodeView(DiGraphView const &g,
                            Node const &removed,
                            Node const &into)
      : g(g), from(removed), to(into) {}

  std::unordered_set<DirectedEdge>
      query_edges(DirectedEdgeQuery const &) const override;
  std::unordered_set<Node> query_nodes(NodeQuery const &) const override;

  ContractNodeView *clone() const override;

private:
  DirectedEdge fix_edge(DirectedEdge const &) const;

private:
  DiGraphView g;
  Node from, to;
};


DiGraphView
    contract_node(DiGraphView const &g, Node const &from, Node const &into);

} // namespace FlexFlow

#endif
