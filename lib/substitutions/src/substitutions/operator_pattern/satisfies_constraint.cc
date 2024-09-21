#include "substitutions/operator_pattern/satisfies_constraint.h"
#include "substitutions/operator_pattern/operator_attribute_expr.h"

namespace FlexFlow {

bool operator_satisfies_constraint(
    PCGOperatorAttrs const &attrs,
    OperatorAttributeConstraint const &constraint) {
  std::optional<OperatorAttributeValue> expr_val =
      evaluate_attribute_expr(constraint.attribute_expr, attrs);

  if (!expr_val.has_value()) {
    return false;
  }

  switch (constraint.constraint_type) {
    case ConstraintType::EQUAL:
      return expr_val.value() == constraint.attribute_value;
    default:
      throw mk_runtime_error(
          fmt::format("Unknown constraint type {}",
                      static_cast<int>(constraint.constraint_type)));
  }
}

} // namespace FlexFlow
