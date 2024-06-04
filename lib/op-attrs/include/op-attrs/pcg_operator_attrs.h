#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PCG_OPERATOR_ATTRS_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PCG_OPERATOR_ATTRS_H

#include "op-attrs/pcg_operator_attrs.dtg.h"

namespace FlexFlow {

bool is_parallel_op(PCGOperatorAttrs const &);
OperatorType get_op_type(PCGOperatorAttrs const &);

} // namespace FlexFlow

#endif
