#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_GET_INCOMING_TENSOR_ROLES_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_GET_INCOMING_TENSOR_ROLES_H

#include "op-attrs/computation_graph_op_attrs.dtg.h"
#include "op-attrs/incoming_tensor_role.dtg.h"
#include "op-attrs/pcg_operator_attrs.dtg.h"

namespace FlexFlow {

std::vector<IncomingTensorRole>
    get_incoming_tensor_roles(ComputationGraphOpAttrs const &, int num_inputs);
std::vector<IncomingTensorRole>
    get_incoming_tensor_roles(PCGOperatorAttrs const &, int num_inputs);

} // namespace FlexFlow

#endif
