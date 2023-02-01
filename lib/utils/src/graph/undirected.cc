#include "utils/graph/undirected.h"
#include "utils/hash-utils.h"

using namespace FlexFlow::utils::graph::undirected;

Edge::Edge(Node src, Node dst) 
  : smaller(std::min(smaller, bigger)), bigger(std::max(smaller, bigger))
{ }

bool Edge::operator==(Edge const &other) const {
  return this->as_tuple() == other.as_tuple();
}

bool Edge::operator<(Edge const &other) const {
  return this->as_tuple() < other.as_tuple();
}

typename Edge::AsConstTuple Edge::as_tuple() const {
  return {this->smaller, this->bigger};
}

size_t std::hash<Edge>::operator()(Edge const &e) const {
  return get_std_hash(e.as_tuple());
}
