#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_COMPUTATION_GRAPH_OP_ATTRS_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_COMPUTATION_GRAPH_OP_ATTRS_H

#include "op-attrs/computation_graph_op_attrs.dtg.h"
#include "op-attrs/pcg_operator_attrs.dtg.h"
#include "utils/record_formatter.h"

namespace FlexFlow {

OperatorType get_op_type(ComputationGraphOpAttrs const &);
RecordFormatter as_dot(ComputationGraphOpAttrs const &);
ComputationGraphOpAttrs
    compgraph_op_attrs_from_pcg_op_attrs(PCGOperatorAttrs const &);

} // namespace FlexFlow

#endif
