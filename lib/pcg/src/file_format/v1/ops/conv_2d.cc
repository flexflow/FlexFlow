#include "pcg/file_format/v1/ops/conv_2d.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1Conv2DAttrs to_v1(Conv2DAttrs const &a) {
  return {a.out_channels,
          a.kernel_h,
          a.kernel_w,
          a.stride_h,
          a.stride_w,
          a.padding_h,
          a.padding_w,
          a.groups,
          to_v1<V1Activation>(a.activation),
          a.use_bias};
}

} // namespace FlexFlow
