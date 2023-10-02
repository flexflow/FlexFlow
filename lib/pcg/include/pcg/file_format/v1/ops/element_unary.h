#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_ELEMENTARY_UNARY_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_ELEMENTARY_UNARY_ATTRS_H

#include "op-attrs/op.h"
#include "op-attrs/ops/element_unary.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1ElementScalarUnaryAttrs {
  req<Op> op;
  req<float> scalar;
};
FF_VISITABLE_STRUCT(V1ElementScalarUnaryAttrs, op, scalar);
CHECK_IS_JSONABLE(V1ElementScalarUnaryAttrs);

V1ElementScalarUnaryAttrs to_v1(ElementScalarUnaryAttrs const &attrs);

struct V1ElementUnaryAttrs {
  req<Op> op;
};
FF_VISITABLE_STRUCT(V1ElementUnaryAttrs, op);
CHECK_IS_JSONABLE(V1ElementUnaryAttrs);

V1ElementUnaryAttrs to_v1(ElementUnaryAttrs const &attrs);

} // namespace FlexFlow

#endif
