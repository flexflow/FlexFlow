#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_ELEMENT_BINARY_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_ELEMENT_BINARY_ATTRS_H

#include "op-attrs/ops/element_binary.h"
#include "pcg/file_format/v1/datatype.h"
#include "pcg/file_format/v1/op.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1ElementBinaryAttrs {
  req<V1Op> type;
  req<V1DataType> compute_type;
  req<bool> should_broadcast_lhs;
  req<bool> should_broadcast_rhs;
};
FF_VISITABLE_STRUCT(V1ElementBinaryAttrs,
                    type,
                    compute_type,
                    should_broadcast_lhs,
                    should_broadcast_rhs);
CHECK_IS_JSONABLE(V1ElementBinaryAttrs);

V1ElementBinaryAttrs to_v1(ElementBinaryAttrs const &a);
ElementBinaryAttrs from_v1(V1ElementBinaryAttrs const &va);

} // namespace FlexFlow

#endif
