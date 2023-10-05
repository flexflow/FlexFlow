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

Conv2DAttrs from_v1(V1Conv2DAttrs const &va) {
  return {va.out_channels,
          va.kernel_h,
          va.kernel_w,
          va.stride_h,
          va.stride_w,
          va.padding_h,
          va.padding_w,
          va.groups,
          from_v1<Activation>(va.activation),
          va.use_bias};
}

} // namespace FlexFlow
