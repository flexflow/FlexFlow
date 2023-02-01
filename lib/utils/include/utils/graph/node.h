#ifndef _FLEXFLOW_UTILS_GRAPH_NODE_H
#define _FLEXFLOW_UTILS_GRAPH_NODE_H

#include <cstddef>
#include <functional>
#include <unordered_set>
#include "tl/optional.hpp"

namespace FlexFlow {
namespace utils {
namespace graph {

struct Node {
public:
  Node() = delete;
  explicit Node(std::size_t idx); 

  bool operator==(Node const &) const;
  bool operator<(Node const &) const;

  using AsConstTuple = std::tuple<size_t>;
  AsConstTuple as_tuple() const;
public:
  std::size_t idx;
};

}
}
}

namespace std {
template <>
struct hash<::FlexFlow::utils::graph::Node> {
  std::size_t operator()(::FlexFlow::utils::graph::Node const &) const;
};
}

namespace FlexFlow {
namespace utils {
namespace graph {

struct NodeQuery {
  tl::optional<std::unordered_set<std::size_t>> nodes;
};

}
}
}



#endif 
