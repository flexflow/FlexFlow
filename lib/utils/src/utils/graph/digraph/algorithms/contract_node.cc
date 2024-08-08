#include "utils/graph/digraph/algorithms/contract_node.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

std::unordered_set<DirectedEdge>
    ContractNodeView::query_edges(DirectedEdgeQuery const &q) const {
  return transform(g.query_edges(q), [&](DirectedEdge const &e) {
    DirectedEdge result = e;
    if (result.src == this->from) {
      result.src = this->to;
    }
    if (result.dst == this->from) {
      result.dst = this->to;
    }
    return result;
  });
}

std::unordered_set<Node>
    ContractNodeView::query_nodes(NodeQuery const &q) const {
  return transform(g.query_nodes(q), [&](Node const &n) {
    if (n == this->from) {
      return this->to;
    } else {
      return n;
    }
  });
}

ContractNodeView *ContractNodeView::clone() const {
  return new ContractNodeView(this->g, this->from, this->to);
}

DiGraphView
    contract_node(DiGraphView const &g, Node const &from, Node const &into) {
  return DiGraphView::create<ContractNodeView>(g, from, into);
}

} // namespace FlexFlow
