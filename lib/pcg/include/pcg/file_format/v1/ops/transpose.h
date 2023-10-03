#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_TRANSPOSE_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_TRANSPOSE_ATTRS_H

#include "op-attrs/ops/transpose.h"
#include "pcg/file_format/v1/ff_dim.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1TransposeAttrs {
  // The size of this vector must be <= MAX_TENSOR_DIMS
  req<std::vector<int>> perm;
};
FF_VISITABLE_STRUCT(V1TransposeAttrs, perm);
CHECK_IS_JSONABLE(V1TransposeAttrs);

V1TransposeAttrs to_v1(TransposeAttrs const &attrs);

} // namespace FlexFlow

#endif
