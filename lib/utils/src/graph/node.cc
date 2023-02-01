#include "utils/graph/node.h"
#include "utils/hash-utils.h"

using namespace FlexFlow::utils::graph;

Node::Node(std::size_t idx) 
  : idx(idx) 
{ }

bool Node::operator==(Node const &other) const {
  return this->as_tuple() == other.as_tuple();
}

bool Node::operator<(Node const &other) const {
  return this->as_tuple() < other.as_tuple();
}

typename Node::AsConstTuple Node::as_tuple() const {
  return {this->idx};
}

namespace std {
std::size_t hash<Node>::operator()(Node const &n) const {
  return get_std_hash(n.as_tuple());
}
}
