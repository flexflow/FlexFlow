#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_REGULARIZER_ATTRS_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_REGULARIZER_ATTRS_H

#include "op-attrs/l1_regularizer_attrs.h"
#include "op-attrs/l2_regularizer_attrs.h"
#include "utils/json.h"

namespace FlexFlow {

using RegularizerAttrs = std::variant<L1RegularizerAttrs, L2RegularizerAttrs>;

CHECK_IS_JSONABLE(RegularizerAttrs);

} // namespace FlexFlow

#endif
