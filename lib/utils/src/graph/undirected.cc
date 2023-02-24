#include "utils/graph/undirected.h"
#include "utils/hash-utils.h"

namespace FlexFlow {
namespace utils {

UndirectedEdge::UndirectedEdge(Node src, Node dst) 
  : smaller(std::min(smaller, bigger)), bigger(std::max(smaller, bigger))
{ }

bool UndirectedEdge::operator==(UndirectedEdge const &other) const {
  return this->as_tuple() == other.as_tuple();
}

bool UndirectedEdge::operator<(UndirectedEdge const &other) const {
  return this->as_tuple() < other.as_tuple();
}

typename UndirectedEdge::AsConstTuple UndirectedEdge::as_tuple() const {
  return {this->smaller, this->bigger};
}

UndirectedEdgeQuery::UndirectedEdgeQuery(tl::optional<std::unordered_set<Node>> const &nodes) 
  : nodes(nodes)
{ }

}
}

namespace std {
using ::FlexFlow::utils::UndirectedEdge;

size_t std::hash<UndirectedEdge>::operator()(UndirectedEdge const &e) const {
  return get_std_hash(e.as_tuple());
}
}
