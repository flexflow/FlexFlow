#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_CONCAT_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_CONCAT_ATTRS_H

#include "op-attrs/ops/concat.h"
#include "pcg/file_format/v1/ff_dim.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1ConcatAttrs {
  req<int> axis;
};
FF_VISITABLE_STRUCT(V1ConcatAttrs, axis);
CHECK_IS_JSONABLE(V1ConcatAttrs);

V1ConcatAttrs to_v1(ConcatAttrs const &attrs);

} // namespace FlexFlow

#endif
