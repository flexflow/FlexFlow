#ifndef _FLEXFLOW_CONV_2D_ATTRS_H
#define _FLEXFLOW_CONV_2D_ATTRS_H

#include "core.h"
#include "op-attrs/activation.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct Conv2DAttrs {
  req<int> out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h,
      padding_w, groups;
  req<optional<Activation>> activation;
  req<bool> use_bias;
};

FF_VISITABLE_STRUCT(Conv2DAttrs,
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
CHECK_VALID_OP_ATTR(Conv2DAttrs);

TensorShape get_kernel_shape(Conv2DAttrs const &, TensorShape const &);
TensorShape get_bias_shape(Conv2DAttrs const &, TensorShape const &);

} // namespace FlexFlow

#endif
