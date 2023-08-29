#ifndef _FLEXFLOW_SUBSTITUTIONS_OUTPUT_GRAPH_H
#define _FLEXFLOW_SUBSTITUTIONS_OUTPUT_GRAPH_H

#include "utils/graph.h"

namespace FlexFlow {

using GraphAttributeValue =
    variant<int, float, bool, std::vector<int>, OperatorType, Activation>;

// NOTE(@wmdi) I am not sure whether these should be part of attribute expr.
struct NodeAttrAccess {
  Node node;
  AttributeExpr<OperatorAttributeKey> attr_expr;
};

struct EdgeAttrAccess {
  OpenMultiDiEdge edge;
  AttributeExpr<TensorAttributeKey> attr_expr;
};

struct AttrConstant {
  GraphAttributeValue value;
};

using GraphAttributeExpr =
    variant<NodeAttrAccess, EdgeAttrAccess, AttrConstant>;

// NOTE(@wmdi): Not sure if it aligns with other design. Or alternatively we can
// define the assignment for each operator type.
struct OperatorAttrAssignment {
  std::unordered_map<OperatorAttributeKey, GraphAttributeExpr> assignment;
};

struct ParallelTensorAttrAssignment {
  std::unordered_map<TensorAttributeKey, GraphAttributeExpr> assignment;
};

struct OutputGraphExpr
    : public strong_typedef<
          OutputGraphExpr,
          OutputLabelledOpenMultiDiGraph<OperatorAttrAssignment,
                                         ParallelTensorAttrAssignment>> {
  using strong_typedef::strong_typedef;
};

} // namespace FlexFlow

#endif
