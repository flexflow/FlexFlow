#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_POOL_2D_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_POOL_2D_ATTRS_H

#include "op-attrs/ops/pool_2d.h"
#include "pcg/file_format/v1/activation.h"
#include "utils/json.h"
#include "utils/visitable.h"

namespace FlexFlow {

enum class V1PoolOp {
  MAX,
  AVG,
};

NLOHMANN_JSON_SERIALIZE_ENUM(V1PoolOp,
                             {{V1PoolOp::MAX, "MAX"}, {V1PoolOp::AVG, "AVG"}});

V1PoolOp to_v1(PoolOp const &op);

struct V1Pool2DAttrs {
  req<int> kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;
  req<V1PoolOp> pool_type;
  req<V1Activation> activation;
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
