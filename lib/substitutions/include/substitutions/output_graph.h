#ifndef _FLEXFLOW_SUBSTITUTIONS_OUTPUT_GRAPH_H
#define _FLEXFLOW_SUBSTITUTIONS_OUTPUT_GRAPH_H

#include "utils/graph.h"

namespace FlexFlow {

// NOTE(@wmdi) I am not sure whether these should be part of attribute expr.
struct OperatorAttrAccess {
  Node node;
  AttributeExpr<OperatorAttributeKey> attr_expr;
};

struct AttrConstant {
  OperatorAttributeValue value;
};

using OperatorAttributeExpr = variant<OperatorAttrAccess, AttrConstant>;

// NOTE(@wmdi): Not sure if it aligns with other design. Or alternatively we can
// define the assignment for each operator type.
struct OperatorAttrAssignment {
  std::unordered_map<OperatorAttributeKey, OperatorAttributeExpr> assignments;
};

struct OutputGraphExpr
    : public strong_typedef<
          OutputGraphExpr,
          NodeLabelledOpenMultiDiGraph<OperatorAttrAssignment>> {
  using strong_typedef::strong_typedef;
};

} // namespace FlexFlow

#endif
