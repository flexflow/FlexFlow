#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PCG_OPERATOR_ATTRS_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PCG_OPERATOR_ATTRS_H

#include "op-attrs/computation_graph_op_attrs.h"
#include "op-attrs/pcg_operator_attrs.dtg.h"

namespace FlexFlow {

bool is_parallel_op(PCGOperatorAttrs const &);
OperatorType get_op_type(PCGOperatorAttrs const &);
ComputationGraphOpAttrs
    compgraph_op_attrs_from_pcg_op_attrs(PCGOperatorAttrs const &);
RecordFormatter as_dot(PCGOperatorAttrs const &);

} // namespace FlexFlow

#endif
