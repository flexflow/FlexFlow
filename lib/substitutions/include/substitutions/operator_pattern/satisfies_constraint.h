#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_OPERATOR_PATTERN_SATISFIES_CONSTRAINT_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_OPERATOR_PATTERN_SATISFIES_CONSTRAINT_H

#include "op-attrs/pcg_operator_attrs.dtg.h"
#include "substitutions/operator_pattern/operator_attribute_constraint.dtg.h"

namespace FlexFlow {

bool operator_satisfies_constraint(
    PCGOperatorAttrs const &params,
    OperatorAttributeConstraint const &constraint);

} // namespace FlexFlow

#endif
