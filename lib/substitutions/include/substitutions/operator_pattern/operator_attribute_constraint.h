#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_OPERATOR_PATTERN_OPERATOR_ATTRIBUTE_CONSTRAINT_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_OPERATOR_PATTERN_OPERATOR_ATTRIBUTE_CONSTRAINT_H

#include "substitutions/operator_pattern/operator_attribute_constraint.dtg.h"

namespace FlexFlow {

OperatorAttributeConstraint op_type_equals_constraint(OperatorType);

OperatorAttributeConstraint op_attr_key_equals(OperatorAttributeKey,
                                               OperatorAttributeValue const &);
OperatorAttributeConstraint
    make_equals_constraint(OperatorAttributeExpr const &,
                           OperatorAttributeValue const &);

} // namespace FlexFlow

#endif
