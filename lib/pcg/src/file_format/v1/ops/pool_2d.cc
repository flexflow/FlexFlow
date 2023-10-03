#include "pcg/file_format/v1/ops/pool_2d.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1PoolOp to_v1(PoolOp const &op) {
  NOT_IMPLEMENTED();
}

V1Pool2DAttrs to_v1(Pool2DAttrs const &a) {
  return {to_v1(a.kernel_h),
          to_v1(a.kernel_w),
          to_v1(a.stride_h),
          to_v1(a.stride_w),
          to_v1(a.padding_h),
          to_v1(a.padding_w),
          to_v1(a.pool_type),
          to_v1(a.activation)};
}

} // namespace FlexFlow
