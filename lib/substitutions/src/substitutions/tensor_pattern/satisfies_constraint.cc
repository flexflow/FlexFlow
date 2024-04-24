#include "substitutions/tensor_pattern/satisfies_constraint.h"
#include "substitutions/tensor_pattern/tensor_attribute_expr.h"

namespace FlexFlow {

bool parallel_tensor_satisfies_constraint(ParallelTensorAttrs const &attrs, TensorAttributeConstraint const &constraint) {
  TensorAttributeValue expr_val = evaluate_attribute_expr(attrs, constraint.attribute_expr);

  switch (constraint.constraint_type) {
    case ConstraintType::EQUAL:
      return expr_val == constraint.attribute_value;
    default:
      throw mk_runtime_error(fmt::format("Unknown constraint type {}", static_cast<int>(constraint.constraint_type)));
  }
}

} // namespace FlexFlow
