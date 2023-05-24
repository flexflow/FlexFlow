#ifndef _FLEXFLOW_CONV_2D_ATTRS_H
#define _FLEXFLOW_CONV_2D_ATTRS_H

#include "op-attrs/activation.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"
#include "op-attrs/tensor_shape.h"
#include "core.h"

namespace FlexFlow {

struct Conv2DAttrs : public use_visitable_cmp<Conv2DAttrs> {
public:
  Conv2DAttrs() = delete;
  Conv2DAttrs(int out_channels, int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h, int padding_w, int groups, optional<Activation> activation, bool use_bias);
public:
  int out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h,
      padding_w, groups;
  optional<Activation> activation;
  bool use_bias;
};

TensorShape get_kernel_shape(Conv2DAttrs const &, TensorShape const &);
TensorShape get_bias_shape(Conv2DAttrs const &, TensorShape const &);

}

VISITABLE_STRUCT(::FlexFlow::Conv2DAttrs, out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, groups, activation, use_bias);
MAKE_VISIT_HASHABLE(::FlexFlow::Conv2DAttrs);

namespace FlexFlow {
static_assert(is_valid_opattr<Conv2DAttrs>::value, "Conv2DAttrs must be a valid opattr (see core.h)");
}

#endif 
