#ifndef _FLEXFLOW_POOL_2D_ATTRS_H
#define _FLEXFLOW_POOL_2D_ATTRS_H

#include "op-attrs/activation.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

enum class PoolOp {
  MAX,
  AVG,
};

struct Pool2DAttrs : use_visitable_cmp<Pool2DAttrs> {
public:
  Pool2DAttrs() = delete;
  Pool2DAttrs(int kernel_h, int kernel_w, int stride_h, int stride_w,
              int padding_h, int padding_w, PoolOp pool_type,
              Activation activation);

public:
  int kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;
  PoolOp pool_type;
  Activation activation;
};

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::Pool2DAttrs, kernel_h, kernel_w, stride_h,
                 stride_w, padding_h, padding_w, pool_type, activation);
MAKE_VISIT_HASHABLE(::FlexFlow::Pool2DAttrs);

namespace fmt {

template <> struct formatter<::FlexFlow::PoolOp> : formatter<string_view> {
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
