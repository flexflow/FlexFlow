#include "utils/graph/undirected.h"
#include "utils/visitable_funcs.h"

namespace FlexFlow {

UndirectedEdge::UndirectedEdge(Node src, Node dst) 
  : smaller(std::min(smaller, bigger)), bigger(std::max(smaller, bigger))
{ }

bool UndirectedEdge::operator==(UndirectedEdge const &other) const {
  return visit_eq(*this, other);
}

bool UndirectedEdge::operator<(UndirectedEdge const &other) const {
  return visit_eq(*this, other);
}

UndirectedEdgeQuery::UndirectedEdgeQuery(tl::optional<std::unordered_set<Node>> const &nodes) 
  : nodes(nodes)
{ }

}

namespace std {
using ::FlexFlow::UndirectedEdge;

size_t std::hash<UndirectedEdge>::operator()(UndirectedEdge const &e) const {
  return visit_hash(e);
}
}
