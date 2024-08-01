#include "substitutions/output_graph/output_operator_attribute_expr.h"
#include "utils/overload.h"
#include "substitutions/operator_pattern/operator_attribute_expr.h"

namespace FlexFlow {

OperatorAttributeValue evaluate_output_operator_attribute_expr(OutputOperatorAttributeExpr const &expr,
                                                               std::unordered_map<PatternNode, PCGOperatorAttrs> const &node_match) {
  return expr.visit<OperatorAttributeValue>(overload {
    [&](OutputOperatorAttrAccess const &a) { 
      return evaluate_attribute_expr(a.attr_expr, node_match.at(a.node)).value();
    },
    [](AttrConstant const &c) { return c.value; },
  });
}

} // namespace FlexFlow
