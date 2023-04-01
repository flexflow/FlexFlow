#include "utils/graph/digraph.h"
#include "utils/visitable_funcs.h"

namespace FlexFlow {

DirectedEdge::DirectedEdge(Node src, Node dst) 
  : src(src), dst(dst)
{ }

bool DirectedEdge::operator==(DirectedEdge const &other) const {
  return visit_eq(*this, other);
}

bool DirectedEdge::operator<(DirectedEdge const &other) const {
  return visit_lt(*this, other);
}

std::ostream &operator<<(std::ostream &s, DirectedEdge const &e) {
  return (
    s << "DirectedEdge{" << e.src.idx << " -> " << e.dst.idx << "}"
  );
}

DirectedEdgeQuery::DirectedEdgeQuery(tl::optional<std::unordered_set<Node>> const &srcs, tl::optional<std::unordered_set<Node>> const &dsts) 
  : srcs(srcs), dsts(dsts)
{ }

}

namespace std {
using ::FlexFlow::DirectedEdge;

size_t std::hash<DirectedEdge>::operator()(DirectedEdge const &e) const {
  return visit_hash(e);
}
}
