#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_LINEAR_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_LINEAR_ATTRS_H

#include "op-attrs/ops/linear.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/file_format/v1/activation.h"
#include "pcg/file_format/v1/datatype.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1L1RegularizerAttrs {
  req<float> lambda;
};
FF_VISITABLE_STRUCT(V1L1RegularizerAttrs, lambda);
CHECK_IS_JSONABLE(V1L1RegularizerAttrs);

V1L1RegularizerAttrs to_v1(L1RegularizerAttrs const &attrs);

struct V1L2RegularizerAttrs {
  req<float> lambda;
};
FF_VISITABLE_STRUCT(V1L2RegularizerAttrs, lambda);
CHECK_IS_JSONABLE(V1L2RegularizerAttrs);

V1L2RegularizerAttrs to_v1(L2RegularizerAttrs const &attrs);

using V1RegularizerAttrs = variant<V1L1RegularizerAttrs, V1L2RegularizerAttrs>;

struct V1LinearAttrs {
  req<int> out_channels;
  req<bool> use_bias;
  req<V1DataType> data_type;
  req<V1Activation> activation;
  req<optional<V1RegularizerAttrs>> regularizer;
};
FF_VISITABLE_STRUCT(
    V1LinearAttrs, out_channels, use_bias, data_type, activation, regularizer);
CHECK_IS_JSONABLE(V1LinearAttrs);

V1LinearAttrs to_v1(LinearAttrs const &attrs);

} // namespace FlexFlow

#endif
