#ifndef _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_LOSS_FUNCTIONS_H
#define _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_LOSS_FUNCTIONS_H

#include "op-attrs/ops/core.h"
#include "op-attrs/ops/loss_functions/loss_function.dtg.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"

namespace FlexFlow {

CHECK_VALID_OP_ATTR(LossAttrs);

LossFunction parse_loss_function_name(std::string const &);

LossFunction get_loss_function(OtherLossAttrs const &);
LossFunction get_loss_function(SparseCategoricalCrossEntropyLossAttrs const &);
LossFunction get_loss_function(LossAttrs const &);

} // namespace FlexFlow

#endif
