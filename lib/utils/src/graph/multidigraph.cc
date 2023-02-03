#include "utils/graph/multidigraph.h"
#include "utils/hash-utils.h"

using namespace FlexFlow::utils::graph::multidigraph;

namespace FlexFlow {
namespace utils {
namespace graph {
namespace multidigraph {

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

std::ostream &operator<<(std::ostream &s, Edge const &e) {
  return (s << "Edge<" << e.src.idx << ":" << e.srcIdx << " -> " << e.dst.idx << ":" << e.dstIdx << ">");
}

EdgeQuery EdgeQuery::with_src_nodes(std::unordered_set<Node> const &nodes) const {
  EdgeQuery e{*this};
  if (e.srcs != tl::nullopt) {
    throw std::runtime_error("expected srcs == tl::nullopt");
  }
  e.srcs = nodes;
  return e;
}

EdgeQuery EdgeQuery::with_src_node(Node const &n) const {
  return this->with_src_nodes({n});
}

EdgeQuery EdgeQuery::with_dst_nodes(std::unordered_set<Node> const &nodes) const {
  EdgeQuery e{*this};
  if (e.dsts != tl::nullopt) {
    throw std::runtime_error("expected dsts == tl::nullopt");
  }
  e.dsts = nodes;
  return e;
}

EdgeQuery EdgeQuery::with_dst_node(Node const &n) const {
  return this->with_dst_nodes({n});
}

EdgeQuery EdgeQuery::with_src_idxs(std::unordered_set<std::size_t> const &idxs) const {
  EdgeQuery e{*this};
  if (e.srcIdxs != tl::nullopt) {
    throw std::runtime_error("expected srcIdxs == tl::nullopt");
  }
  e.srcIdxs = idxs;
  return e;
}

EdgeQuery EdgeQuery::with_src_idx(std::size_t idx) const {
  return this->with_src_idxs({idx});
}

EdgeQuery EdgeQuery::with_dst_idxs(std::unordered_set<std::size_t> const &idxs) const {
  EdgeQuery e{*this};
  if (e.dstIdxs != tl::nullopt) {
    throw std::runtime_error("expected dstIdxs == tl::nullopt");
  }
  e.dstIdxs = idxs;
  return e;
}

EdgeQuery EdgeQuery::with_dst_idx(std::size_t idx) const {
  return this->with_dst_idxs({idx});
}

EdgeQuery EdgeQuery::all() {
  return EdgeQuery{};
}

}
}
}
}

namespace std {
std::size_t hash<Edge>::operator()(Edge const &e) const {
  return get_std_hash(e.as_tuple());
}
}
