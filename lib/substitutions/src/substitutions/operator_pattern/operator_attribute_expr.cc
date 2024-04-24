#include "substitutions/operator_pattern/operator_attribute_expr.h"
#include "substitutions/operator_pattern/get_attribute.h"
#include "substitutions/operator_pattern/eval_list_access.h"
#include "substitutions/operator_pattern/eval_list_size.h"
#include "utils/overload.h"

namespace FlexFlow {

std::optional<OperatorAttributeValue>
    evaluate_attribute_expr(PCGOperatorAttrs const &attrs,
                            OperatorAttributeExpr const &expr) {
  return expr.visit<
    std::optional<OperatorAttributeValue>
  >(overload {
    [&](OperatorAttributeKey const &k) { return get_attribute(attrs, k); },
    [&](OperatorAttributeListSize const &k) { return eval_list_size(attrs, k); },
    [&](OperatorAttributeListIndexAccess const &k) { return eval_list_access(attrs, k); },
  });
}

} // namespace FlexFlow
