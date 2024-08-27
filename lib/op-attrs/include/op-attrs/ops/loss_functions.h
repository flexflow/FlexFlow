#ifndef _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_LOSS_FUNCTIONS_H
#define _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_LOSS_FUNCTIONS_H

#include "core.h"
#include "loss_attrs.dtg.h"
#include "loss_function.dtg.h"
#include "other_loss_attrs.dtg.h"
#include "sparse_categorical_ce_loss_attrs.dtg.h"

namespace FlexFlow {

LossFunction get_loss_function(LossAttrs const &);
LossFunction parse_loss_name(std::string const &raw_name);

} // namespace FlexFlow

#endif
