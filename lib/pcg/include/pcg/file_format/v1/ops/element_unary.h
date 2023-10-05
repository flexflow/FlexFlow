#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_ELEMENTARY_UNARY_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_ELEMENTARY_UNARY_ATTRS_H

#include "op-attrs/ops/element_unary.h"
#include "pcg/file_format/v1/op.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1ElementScalarUnaryAttrs {
  req<V1Op> op;
  req<float> scalar;
};
FF_VISITABLE_STRUCT(V1ElementScalarUnaryAttrs, op, scalar);
CHECK_IS_JSONABLE(V1ElementScalarUnaryAttrs);

V1ElementScalarUnaryAttrs to_v1(ElementScalarUnaryAttrs const &a);
ElementScalarUnaryAttrs from_v1(V1ElementScalarUnaryAttrs const &va);

struct V1ElementUnaryAttrs {
  req<V1Op> op;
};
FF_VISITABLE_STRUCT(V1ElementUnaryAttrs, op);
CHECK_IS_JSONABLE(V1ElementUnaryAttrs);

V1ElementUnaryAttrs to_v1(ElementUnaryAttrs const &a);
ElementUnaryAttrs from_v1(V1ElementUnaryAttrs const &va);

} // namespace FlexFlow

#endif
