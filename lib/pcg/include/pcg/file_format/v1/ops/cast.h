#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_CAST_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_CAST_ATTRS_H

#include "op-attrs/ops/cast.h"
#include "op-attrs/datatype.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1CastAttrs {
  req<DataType> dtype;
};
FF_VISITABLE_STRUCT(V1CastAttrs, dtype);
CHECK_IS_JSONABLE(V1CastAttrs);

V1CastAttrs to_v1(CastAttrs const &attrs);

} // namespace FlexFlow

#endif
