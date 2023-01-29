#ifndef _FLEXFLOW_FFC_EDGE_H
#define _FLEXFLOW_FFC_EDGE_H

#include "node.h"

namespace FlexFlow {
namespace PCG {

struct Edge {
public:
  Edge(Node const &srcOp, Node const &dstOp, int srcIdx, int dstIdx);

  void replace_node(Node const &currentOp, Node const &replaceWith);

  using AsConstTuple = std::tuple<Node, Node, int, int>;
  AsConstTuple as_tuple() const;
public:
  Node srcOp, dstOp;
  int srcIdx, dstIdx;
};

bool operator==(Edge const &lhs, Edge const &rhs);
bool operator<(Edge const &lhs, Edge const &rhs);

}
}

namespace std {
template <>
struct hash<FlexFlow::PCG::Edge> {
  size_t operator()(FlexFlow::PCG::Edge const &e) const;
};
}

#endif
