#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_RESHAPE_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_RESHAPE_ATTRS_H

#include "op-attrs/ops/reshape.h"
#include "pcg/file_format/v1/tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1ReshapeAttrs {
  V1TensorShape shape;
};
FF_VISITABLE_STRUCT(V1ReshapeAttrs, shape);
CHECK_IS_JSONABLE(V1ReshapeAttrs);

V1ReshapeAttrs to_v1(ReshapeAttrs const &a);
ReshapeAttrs from_v1(V1ReshapeAttrs const &va);

} // namespace FlexFlow

#endif
