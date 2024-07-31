#include "substitutions/operator_pattern/operator_attribute_constraint.h"

namespace FlexFlow {

OperatorAttributeConstraint op_type_equals_constraint(OperatorType op_type) {
  return OperatorAttributeConstraint{
    ConstraintType::EQUAL,
    OperatorAttributeExpr{OperatorAttributeKey::OP_TYPE},
    OperatorAttributeValue{op_type},
  };
}

} // namespace FlexFlow
