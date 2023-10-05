#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_CAST_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_CAST_ATTRS_H

#include "op-attrs/ops/cast.h"
#include "pcg/file_format/v1/datatype.h"
#include "utils/json.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1CastAttrs {
  req<V1DataType> dtype;
};
FF_VISITABLE_STRUCT(V1CastAttrs, dtype);
CHECK_IS_JSONABLE(V1CastAttrs);

V1CastAttrs to_v1(CastAttrs const &a);
CastAttrs from_v1(V1CastAttrs const &va);

} // namespace FlexFlow

#endif
