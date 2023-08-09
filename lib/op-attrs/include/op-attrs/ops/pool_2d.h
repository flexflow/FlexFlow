#ifndef _FLEXFLOW_POOL_2D_ATTRS_H
#define _FLEXFLOW_POOL_2D_ATTRS_H

#include "core.h"
#include "op-attrs/activation.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

enum class PoolOp {
  MAX,
  AVG,
};

struct Pool2DAttrs {
  req<int> kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;
  req<PoolOp> pool_type;
  req<Activation> activation;
};
FF_VISITABLE_STRUCT(Pool2DAttrs,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    padding_h,
                    padding_w,
                    pool_type,
                    activation);
CHECK_VALID_OP_ATTR(Pool2DAttrs);

} // namespace FlexFlow

namespace fmt {

template <>
struct formatter<::FlexFlow::PoolOp> : formatter<string_view> {
  template <typename FormatContext>
  auto format(::FlexFlow::PoolOp o, FormatContext &ctx) const
      -> decltype(ctx.out()) {
    using namespace FlexFlow;

    string_view name = "unknown";
    switch (o) {
      case PoolOp::AVG:
        name = "Avg";
        break;
      case PoolOp::MAX:
        name = "Max";
        break;
    }
    return formatter<string_view>::format(name, ctx);
  }
};

} // namespace fmt

#endif
