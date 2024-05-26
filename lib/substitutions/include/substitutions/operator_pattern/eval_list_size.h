#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_OPERATOR_PATTERN_OPERATOR_ATTRIBUTE_LIST_SIZE_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_OPERATOR_PATTERN_OPERATOR_ATTRIBUTE_LIST_SIZE_H

#include "op-attrs/pcg_operator_attrs.dtg.h"
#include "substitutions/operator_pattern/operator_attribute_list_size.dtg.h"
#include "substitutions/operator_pattern/operator_attribute_value.dtg.h"

namespace FlexFlow {

std::optional<OperatorAttributeValue>
    eval_list_size(PCGOperatorAttrs const &attrs,
                   OperatorAttributeListSize const &acc);

} // namespace FlexFlow

#endif
