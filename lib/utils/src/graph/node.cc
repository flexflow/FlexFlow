#include "utils/graph/node.h"
#include "utils/hash-utils.h"
#include <sstream>

namespace FlexFlow {
namespace utils {
namespace graph {

Node::Node(std::size_t idx) 
  : idx(idx) 
{ }

bool Node::operator==(Node const &other) const {
  return this->as_tuple() == other.as_tuple();
}

bool Node::operator!=(Node const &other) const {
  return this->as_tuple() != other.as_tuple();
}

bool Node::operator<(Node const &other) const {
  return this->as_tuple() < other.as_tuple();
}

typename Node::AsConstTuple Node::as_tuple() const {
  return {this->idx};
}

std::string Node::to_string() const {
  std::ostringstream oss;
  oss << *this;
  return oss.str();
}

std::ostream &operator<<(std::ostream &s, Node const &n) {
  s << "Node(" << n.idx << ")";
  return s;
}

NodeQuery::NodeQuery(std::unordered_set<Node> const &nodes)
  : NodeQuery(tl::optional<std::unordered_set<Node>>{nodes})
{ }

NodeQuery::NodeQuery(tl::optional<std::unordered_set<Node>> const &nodes) 
  : nodes(nodes)
{ }

}
}
}

namespace std {
using ::FlexFlow::utils::graph::Node;

std::size_t hash<Node>::operator()(Node const &n) const {
  return get_std_hash(n.as_tuple());
}
}
