#include "utils/graph/digraph.h"
#include "utils/hash-utils.h"

namespace FlexFlow {
namespace utils {
namespace graph {
namespace digraph {

Edge::Edge(Node src, Node dst) 
  : src(src), dst(dst)
{ }

bool Edge::operator==(Edge const &other) const {
  return this->as_tuple() == other.as_tuple();
}

bool Edge::operator<(Edge const &other) const {
  return this->as_tuple() < other.as_tuple();
}

typename Edge::AsConstTuple Edge::as_tuple() const {
  return {this->src, this->dst};
}

std::ostream &operator<<(std::ostream &s, Edge const &e) {
  return (
    s << "Edge{" << e.src.idx << " -> " << e.dst.idx << "}"
  );
}

EdgeQuery::EdgeQuery(tl::optional<std::unordered_set<Node>> const &srcs, tl::optional<std::unordered_set<Node>> const &dsts) 
  : srcs(srcs), dsts(dsts)
{ }

}
}
}
}

namespace std {
using ::FlexFlow::utils::graph::digraph::Edge;

size_t std::hash<Edge>::operator()(Edge const &e) const {
  return get_std_hash(e.as_tuple());
}
}
