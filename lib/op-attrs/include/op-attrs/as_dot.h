#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_AS_DOT_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_AS_DOT_H

#include "op-attrs/computation_graph_op_attrs.dtg.h"
#include "op-attrs/pcg_operator_attrs.dtg.h"
#include "utils/record_formatter.h"

namespace FlexFlow {

RecordFormatter as_dot(ComputationGraphOpAttrs const &);
RecordFormatter as_dot(PCGOperatorAttrs const &);

} // namespace FlexFlow

#endif
