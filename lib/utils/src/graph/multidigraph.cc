#include "utils/graph/multidigraph.h"
#include "utils/hash-utils.h"

using namespace FlexFlow::utils::graph::multidigraph;

Edge::Edge(Node src, Node dst, size_t srcIdx, size_t dstIdx)
  : src(src), dst(dst), srcIdx(srcIdx), dstIdx(dstIdx)
{ }

typename Edge::AsConstTuple Edge::as_tuple() const {
  return {this->src, this->dst, this->srcIdx, this->dstIdx};
}

bool Edge::operator==(Edge const &other) const {
  return this->as_tuple() == other.as_tuple();
}

bool Edge::operator<(Edge const &other) const {
  return this->as_tuple() < other.as_tuple();
}

namespace std {
std::size_t hash<Edge>::operator()(Edge const &e) const {
  return get_std_hash(e.as_tuple());
}
}
