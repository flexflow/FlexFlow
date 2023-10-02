#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_ELEMENT_BINARY_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_ELEMENT_BINARY_ATTRS_H

#include "op-attrs/datatype.h"
#include "op-attrs/op.h"
#include "op-attrs/ops/element_binary.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1ElementBinaryAttrs {
  req<Op> type;
  req<DataType> compute_type;
  req<bool> should_broadcast_lhs;
  req<bool> should_broadcast_rhs;
};
FF_VISITABLE_STRUCT(V1ElementBinaryAttrs,
                    type,
                    compute_type,
                    should_broadcast_lhs,
                    should_broadcast_rhs);
CHECK_IS_JSONABLE(V1ElementBinaryAttrs);

V1ElementBinaryAttrs to_v1(ElementBinaryAttrs const &attrs);

} // namespace FlexFlow

#endif
