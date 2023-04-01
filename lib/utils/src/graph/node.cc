#include "utils/graph/node.h"
#include <sstream>
#include "utils/visitable_funcs.h"

namespace FlexFlow {

Node::Node(std::size_t idx) 
  : idx(idx) 
{ }

bool Node::operator==(Node const &other) const {
  return visit_eq(*this, other);
}

bool Node::operator!=(Node const &other) const {
  return visit_neq(*this, other);
}

bool Node::operator<(Node const &other) const {
  return visit_lt(*this, other);
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

namespace std {
using ::FlexFlow::Node;

std::size_t hash<Node>::operator()(Node const &n) const {
  return visit_hash(n);
}
}
