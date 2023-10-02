#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_CONV_2D_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_CONV_2D_ATTRS_H

#include "op-attrs/activation.h"
#include "op-attrs/ops/conv_2d.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1Conv2DAttrs {
  req<int> out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h,
      padding_w, groups;
  req<optional<Activation>> activation;
  req<bool> use_bias;
};

FF_VISITABLE_STRUCT(V1Conv2DAttrs,
                    out_channels,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    padding_h,
                    padding_w,
                    groups,
                    activation,
                    use_bias);
CHECK_IS_JSONABLE(V1Conv2DAttrs);

V1Conv2DAttrs to_v1(Conv2DAttrs const &attrs);

} // namespace FlexFlow

#endif
