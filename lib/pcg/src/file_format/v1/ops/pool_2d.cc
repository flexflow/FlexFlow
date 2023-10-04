#include "pcg/file_format/v1/ops/pool_2d.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1PoolOp to_v1(PoolOp const &op) {
  // There should be a better way of doing this.
  switch (op) {
    case PoolOp::MAX:
      return V1PoolOp::MAX;
    case PoolOp::AVG:
      return V1PoolOp::AVG;
    default:
      NOT_REACHABLE();
  }
}

V1Pool2DAttrs to_v1(Pool2DAttrs const &a) {
  return {a.kernel_h,
          a.kernel_w,
          a.stride_h,
          a.stride_w,
          a.padding_h,
          a.padding_w,
          to_v1(a.pool_type),
          to_v1(a.activation)};
}

} // namespace FlexFlow
