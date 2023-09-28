#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPERATOR_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPERATOR_ATTRS_H

#include "op-attrs/operator_attrs.h"
#include "utils/json.h"
#include "utils/variant.h"

namespace FlexFlow {

struct V1Conv2DAttrs {
  req<int> out_channels;
  req<int> kernel_h;
  req<int> kernel_w;
  req<int> stride_h;
  req<int> stride_w;
  req<int> padding_h;
  req<int> padding_w;
  req<int> groups;
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

using V1CompGraphOperatorAttrs = variant<V1Conv2DAttrs>;
using V1PCGOperatorAttrs = variant<V1Conv2DAttrs>;

V1CompGraphOperatorAttrs to_v1(CompGraphOperatorAttrs const &attrs);

} // namespace FlexFlow

#endif
