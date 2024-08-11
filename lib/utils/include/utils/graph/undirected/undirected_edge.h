#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UNDIRECTED_UNDIRECTED_EDGE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UNDIRECTED_UNDIRECTED_EDGE_H

#include "utils/graph/node/node.dtg.h"
namespace FlexFlow {

struct UndirectedEdge {
public:
  UndirectedEdge() = delete;
  UndirectedEdge(Node const &src, Node const &dst);

  bool operator==(UndirectedEdge const &) const;
  bool operator!=(UndirectedEdge const &) const;
  bool operator<(UndirectedEdge const &) const;

public:
  Node smaller;
  Node bigger;
};

bool is_connected_to(UndirectedEdge const &, Node const &);

} // namespace FlexFlow

namespace std {

template <>
struct hash<::FlexFlow::UndirectedEdge> {
  size_t operator()(::FlexFlow::UndirectedEdge const &) const;
};

} // namespace std

namespace FlexFlow {
std::string format_as(UndirectedEdge const &);
std::ostream &operator<<(std::ostream &, UndirectedEdge const &);
} // namespace FlexFlow

#endif
