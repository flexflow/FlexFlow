#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_OP_ATTRS_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_OP_ATTRS_H

#include "op-attrs/parallel_op_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/pcg_operator_attrs.dtg.h"
#include "utils/record_formatter.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(ParallelOpAttrs const &, ParallelTensorShape const &);
PCGOperatorAttrs pcg_op_attrs_from_parallel_op_attrs(ParallelOpAttrs const &);

} // namespace FlexFlow

#endif
