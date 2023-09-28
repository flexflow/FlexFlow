#include "pcg/file_format/v1/operator_attrs.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1Conv2DAttrs to_v1(Conv2DAttrs const &attrs) {
  return {
    attrs.out_channels,
    attrs.kernel_h,
    attrs.kernel_w,
    attrs.stride_h,
    attrs.stride_w,
    attrs.padding_h,
    attrs.padding_w,
    attrs.groups,
    attrs.activation,
    attrs.use_bias,
  };
}

} // namespace FlexFlow
