#ifndef _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_LOSS_FUNCTIONS_H
#define _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_LOSS_FUNCTIONS_H

#include "op-attrs/ops/core.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "op-attrs/ops/loss_functions/loss_function.dtg.h"
#include "op-attrs/ops/loss_functions/nonconfigurable_loss_attrs.dtg.h"
#include "op-attrs/ops/loss_functions/sparse_categorical_cross_entropy_loss_attrs.dtg.h"

namespace FlexFlow {

LossFunction get_loss_function(LossAttrs const &);
LossFunction parse_loss_name(std::string const &raw_name);

} // namespace FlexFlow

#endif
