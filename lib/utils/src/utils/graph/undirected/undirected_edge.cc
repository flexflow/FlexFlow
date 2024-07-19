#include "utils/graph/undirected/undirected_edge.h"
#include "utils/hash/tuple.h"

namespace FlexFlow {

UndirectedEdge::UndirectedEdge(Node const &n1, Node const &n2)
    : smaller(std::min(n1, n2)), bigger(std::max(n1, n2)) {}

static std::tuple<Node const &, Node const &> tie(UndirectedEdge const &e) {
  return std::tie(e.smaller, e.bigger);
}

bool UndirectedEdge::operator==(UndirectedEdge const &other) const {
  return tie(*this) == tie(other);
}

bool UndirectedEdge::operator!=(UndirectedEdge const &other) const {
  return tie(*this) != tie(other);
}

bool UndirectedEdge::operator<(UndirectedEdge const &other) const {
  return tie(*this) < tie(other);
}

bool is_connected_to(UndirectedEdge const &e, Node const &n) {
  return e.bigger == n || e.smaller == n;
}

} // namespace FlexFlow

namespace std {

using namespace FlexFlow;

size_t hash<UndirectedEdge>::operator()(UndirectedEdge const &e) const {
  std::tuple<Node, Node> members = ::FlexFlow::tie(e);
  return std::hash<decltype(members)>{}(members);
}

} // namespace std
