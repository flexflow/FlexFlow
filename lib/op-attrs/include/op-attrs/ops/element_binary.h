#ifndef _FLEXFLOW_ELEMENT_BINARY_ATTRS_H
#define _FLEXFLOW_ELEMENT_BINARY_ATTRS_H

#include "core.h"
#include "op-attrs/datatype.h"
#include "op-attrs/op.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct ElementBinaryAttrs {
  Op type;
  DataType compute_type;
  bool should_broadcast_lhs;
  req<bool> should_broadcast_rhs;
};
FF_VISITABLE_STRUCT(ElementBinaryAttrs,
                    type,
                    compute_type,
                    should_broadcast_lhs,
                    should_broadcast_rhs);
FF_VISIT_FMTABLE(ElementBinaryAttrs);

CHECK_VALID_OP_ATTR(ElementBinaryAttrs);

} // namespace FlexFlow

#endif
