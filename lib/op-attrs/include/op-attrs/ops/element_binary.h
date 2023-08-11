#ifndef _FLEXFLOW_ELEMENT_BINARY_ATTRS_H
#define _FLEXFLOW_ELEMENT_BINARY_ATTRS_H

#include "core.h"
#include "../datatype.h"
#include "../op.h"
#include "../parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct ElementBinaryAttrs {
  req<Op> type;
  req<DataType> compute_type;
  req<bool> should_broadcast_lhs;
  req<bool> should_broadcast_rhs;
};
FF_VISITABLE_STRUCT(ElementBinaryAttrs,
                    type,
                    compute_type,
                    should_broadcast_lhs,
                    should_broadcast_rhs);
CHECK_VALID_OP_ATTR(ElementBinaryAttrs);

} // namespace FlexFlow

#endif
