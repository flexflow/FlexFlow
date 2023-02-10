#include "utils/graph/digraph.h"
#include "utils/hash-utils.h"

namespace FlexFlow {
namespace utils {

DirectedEdge::DirectedEdge(Node src, Node dst) 
  : src(src), dst(dst)
{ }

bool DirectedEdge::operator==(DirectedEdge const &other) const {
  return this->as_tuple() == other.as_tuple();
}

bool DirectedEdge::operator<(DirectedEdge const &other) const {
  return this->as_tuple() < other.as_tuple();
}

typename DirectedEdge::AsConstTuple DirectedEdge::as_tuple() const {
  return {this->src, this->dst};
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
}

namespace std {
using ::FlexFlow::utils::DirectedEdge;

size_t std::hash<DirectedEdge>::operator()(DirectedEdge const &e) const {
  return get_std_hash(e.as_tuple());
}
}
