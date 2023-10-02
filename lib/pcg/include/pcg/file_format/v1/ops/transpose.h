#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_TRANSPOSE_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_TRANSPOSE_ATTRS_H

#include "op-attrs/ff_dim.h"
#include "op-attrs/ops/transpose.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1TransposeAttrs {
  // FIXME: stack_vector is probably causing a problem because it is not
  // automatically (de)serializable, but need to ensure that one way or the
  // other.
  // stack_vector<ff_dim_t, MAX_TENSOR_DIM> axes;
  // req<stack_vector<ff_dim_t, MAX_TENSOR_DIM>> perm;
};
FF_VISITABLE_STRUCT(V1TransposeAttrs// , perm
                    );
CHECK_IS_JSONABLE(V1TransposeAttrs);

V1TransposeAttrs to_v1(TransposeAttrs const &attrs);

} // namespace FlexFlow

#endif
