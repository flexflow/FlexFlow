#include "edge.h"

namespace FlexFlow {
namespace PCG {

Edge::Edge(Node const &srcOp, Node const &dstOp, int srcIdx, int dstIdx)
    : srcOp(srcOp), dstOp(dstOp), srcIdx(srcIdx), dstIdx(dstIdx) {}

typename Edge::AsConstTuple Edge::as_tuple() const {
  return { this->srcOp, this->dstOp, this->srcIdx, this->dstIdx };
}

bool operator==(Edge const &lhs, Edge const &rhs) {
  return lhs.as_tuple() == rhs.as_tuple();
}

bool operator<(Edge const &lhs, Edge const &rhs) {
  return lhs.as_tuple() < rhs.as_tuple();
}

void Edge::replace_node(Node const &currentOp, Node const &replaceWith) {
  if (this->srcOp == currentOp) {
    this->srcOp = replaceWith;
  }
  if (this->dstOp == currentOp) {
    this->dstOp = replaceWith;
  }
}


}
}
