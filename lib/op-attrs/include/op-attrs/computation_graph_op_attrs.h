#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_COMPUTATION_GRAPH_OP_ATTRS_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_COMPUTATION_GRAPH_OP_ATTRS_H

#include "op-attrs/computation_graph_op_attrs.dtg.h"
#include "utils/record_formatter.h"

namespace FlexFlow {

OperatorType get_op_type(ComputationGraphOpAttrs const &);
RecordFormatter as_dot(ComputationGraphOpAttrs const &);

} // namespace FlexFlow

#endif
