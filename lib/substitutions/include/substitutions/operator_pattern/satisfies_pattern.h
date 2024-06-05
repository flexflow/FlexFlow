#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_OPERATOR_PATTERN_SATISFIES_PATTERN_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_OPERATOR_PATTERN_SATISFIES_PATTERN_H

#include "op-attrs/pcg_operator_attrs.dtg.h"
#include "substitutions/operator_pattern/operator_attribute_pattern.dtg.h"

namespace FlexFlow {

bool operator_satisfies_pattern(PCGOperatorAttrs const &attrs,
                                OperatorAttributePattern const &pattern);

} // namespace FlexFlow

#endif
