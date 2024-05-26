#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_OPERATOR_PATTERN_OPERATOR_ATTRIBUTE_LIST_ACCESS_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_OPERATOR_PATTERN_OPERATOR_ATTRIBUTE_LIST_ACCESS_H

#include "op-attrs/pcg_operator_attrs.dtg.h"
#include "substitutions/operator_pattern/operator_attribute_list_access.dtg.h"
#include "substitutions/operator_pattern/operator_attribute_value.dtg.h"
#include <optional>

namespace FlexFlow {

std::optional<OperatorAttributeValue>
    eval_list_access(PCGOperatorAttrs const &attrs,
                     OperatorAttributeListIndexAccess const &);

} // namespace FlexFlow

#endif
