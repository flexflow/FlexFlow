#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_POOL_2D_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_POOL_2D_ATTRS_H

#include "op-attrs/activation.h"
#include "op-attrs/ops/pool_2d.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1Pool2DAttrs {
  req<int> kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;
  req<PoolOp> pool_type;
  req<Activation> activation;
};
FF_VISITABLE_STRUCT(V1Pool2DAttrs,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    padding_h,
                    padding_w,
                    pool_type,
                    activation);
CHECK_IS_JSONABLE(V1Pool2DAttrs);

V1Pool2DAttrs to_v1(Pool2DAttrs const &attrs);

} // namespace FlexFlow

#endif
