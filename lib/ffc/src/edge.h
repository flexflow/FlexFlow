#ifndef _FLEXFLOW_FFC_EDGE_H
#define _FLEXFLOW_FFC_EDGE_H

#include "op-meta/operator_params.h"

namespace FlexFlow {
namespace ffc {

struct Edge {
public:
  Edge(opmeta::OperatorParameters const &srcOp, opmeta::OperatorParameters const &dstOp, int srcIdx, int dstIdx);

  void replace_node(opmeta::OperatorParameters const &currentOp, opmeta::OperatorParameters const &replaceWith);

  using AsConstTuple = std::tuple<opmeta::OperatorParameters, opmeta::OperatorParameters, int, int>;
  AsConstTuple as_tuple() const;
public:
  opmeta::OperatorParameters srcOp, dstOp;
  int srcIdx, dstIdx;
};

bool operator==(Edge const &lhs, Edge const &rhs);
bool operator<(Edge const &lhs, Edge const &rhs);

}
}

namespace std {
template <>
struct hash<::FlexFlow::ffc::Edge> {
  size_t operator()(::FlexFlow::ffc::Edge const &e) const;
};
}

#endif
