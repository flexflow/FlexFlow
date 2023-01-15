#include "edge.h"

namespace FlexFlow {
namespace PCG {

Edge::Edge(Node const &srcOp, Node const &dstOp, int srcIdx, int dstIdx)
    : srcOp(srcOp), dstOp(dstOp), srcIdx(srcIdx), dstIdx(dstIdx) {}

bool Edge::operator==(Edge const &rhs) const {
  if (srcOp != rhs.srcOp) {
    return false;
  }
  if (dstOp != rhs.dstOp) {
    return false;
  }
  if (srcIdx != rhs.srcIdx) {
    return false;
  }
  if (dstIdx != rhs.dstIdx) {
    return false;
  }
  return true;
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
