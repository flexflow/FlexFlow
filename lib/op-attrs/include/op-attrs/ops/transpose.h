#ifndef _FLEXFLOW_OP_META_OPS_TRANSPOSE_ATTRS_H
#define _FLEXFLOW_OP_META_OPS_TRANSPOSE_ATTRS_H

#include "core.h"
#include "op-attrs/ff_dim.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct TransposeAttrs {
  req<stack_vector<ff_dim_t, MAX_TENSOR_DIM>> perm;
};
FF_VISITABLE_STRUCT(TransposeAttrs, perm);
CHECK_VALID_OP_ATTR(TransposeAttrs);

} // namespace FlexFlow

#endif
