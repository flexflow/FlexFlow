#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_FLAT_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_FLAT_ATTRS_H

#include "op-attrs/ops/flat.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1FlatAttrs {};
FF_VISITABLE_STRUCT(V1FlatAttrs);
CHECK_IS_JSONABLE(V1FlatAttrs);

V1FlatAttrs to_v1(FlatAttrs const &attrs);

} // namespace FlexFlow

#endif
