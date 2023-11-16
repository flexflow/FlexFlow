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

PoolOp from_v1(V1PoolOp const &vop) {
  // There should be a better way of doing this.
  switch (vop) {
    case V1PoolOp::MAX:
      return PoolOp::MAX;
    case V1PoolOp::AVG:
      return PoolOp::AVG;
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

Pool2DAttrs from_v1(V1Pool2DAttrs const &va) {
  return {va.kernel_h,
          va.kernel_w,
          va.stride_h,
          va.stride_w,
          va.padding_h,
          va.padding_w,
          from_v1(va.pool_type),
          from_v1(va.activation)};
}

} // namespace FlexFlow
