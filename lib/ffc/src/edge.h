#ifndef _FLEXFLOW_FFC_EDGE_H
#define _FLEXFLOW_FFC_EDGE_H

#include "node.h"

namespace FlexFlow {
namespace PCG {

struct Edge {
  Edge(Node const &_srcOp, Node const &_dstOp, int _srcIdx, int _dstIdx);
  bool operator==(Edge const &rhs) const;
  Node srcOp, dstOp;
  int srcIdx, dstIdx;

  void replace_node(Node const &currentOp, Node const &replaceWith);
};

struct EdgeCompare {
  bool operator()(Edge const &a, Edge const &b) const {
    if (!(a.srcOp == b.srcOp)) {
      return a.srcOp < b.srcOp;
    }
    if (!(a.dstOp == b.dstOp)) {
      return a.dstOp < b.dstOp;
    }
    if (a.srcIdx != b.srcIdx) {
      return a.srcIdx < b.srcIdx;
    }
    if (a.dstIdx != b.dstIdx) {
      return a.dstIdx < b.dstIdx;
    }
    return false;
  };
};

}
}

#endif
