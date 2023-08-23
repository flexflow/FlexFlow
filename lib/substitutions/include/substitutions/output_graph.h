#ifndef _FLEXFLOW_SUBSTITUTIONS_OUTPUT_GRAPH_H
#define _FLEXFLOW_SUBSTITUTIONS_OUTPUT_GRAPH_H

#include "utils/graph.h"

namespace FlexFlow {

using GraphAttributeKey = variant<OperatorAttributeKey, TensorAttributeKey>;
using GraphAttributeValue =
    variant<int, float, bool, std::vector<int>, OperatorType, Activation>;

// NOTE(@wmdi) I am not sure whether these should be part of attribute expr.
struct NodeAttrAccess {
  Node node;
  GraphAttributeKey attr_expr;
};

struct EdgeAttrAccess {
  OpenMultiDiEdge edge;
  GraphAttributeKey attr_expr;
};

struct AttrConstant {
  GraphAttributeValue value;
};

using GraphAttributeExprLeaf =
    variant<NodeAttrAccess, EdgeAttrAccess, AttrConstant>;

enum class AttrOpType { ADD, SUB, MUL, DIV };

struct AttrUnary {
  AttrOpType op_type;
  GraphAttributeExprLeaf lhs;
  GraphAttributeExprLeaf rhs;
};

struct AttrBinary {
  AttrOpType op_type;
  GraphAttributeExprLeaf lhs;
  GraphAttributeExprLeaf rhs;
};

using GraphAttributeExpr =
    variant<AttrUnary, AttrBinary, GraphAttributeExprLeaf>;

// NOTE(@wmdi): Not sure if it aligns with other design. Or alternatively we can
// define the assignment for each operator type.
struct OperatorAttrAssignment {
  std::vector<std::pair<OperatorAttributeKey, GraphAttributeExpr>> assignment;
};

struct ParallelTensorAttrAssignment {
  std::vector<std::pair<TensorAttributeKey, GraphAttributeExpr>> assignment;
};

struct OutputGraph
    : public strong_typedef<
          OutputGraph,
          OutputLabelledMultiDiGraph<OperatorAttrAssignment,
                                     ParallelTensorAttrAssignment>> {
  using strong_typedef::strong_typedef;
};

} // namespace FlexFlow

#endif
