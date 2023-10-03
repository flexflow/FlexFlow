#include "pcg/file_format/v1/ops/conv_2d.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1Conv2DAttrs to_v1(Conv2DAttrs const &a) {
  return {to_v1(a.out_channels),
          to_v1(a.kernel_h),
          to_v1(a.kernel_w),
          to_v1(a.stride_h),
          to_v1(a.stride_w),
          to_v1(a.padding_h),
          to_v1(a.padding_w),
          to_v1(a.groups),
          to_v1<V1Activation>(a.activation),
          to_v1(a.use_bias)};
}

} // namespace FlexFlow
