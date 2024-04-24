#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPERATOR_TYPE_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPERATOR_TYPE_H

#include "op-attrs/operator_type.dtg.h"

namespace FlexFlow {

std::string get_operator_type_name(OperatorType);
bool is_parallel_op(OperatorType);

} // namespace FlexFlow

#endif
