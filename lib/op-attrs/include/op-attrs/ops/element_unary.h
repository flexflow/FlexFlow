#ifndef _FLEXFLOW_ELEMENTARY_UNARY_ATTRS_H
#define _FLEXFLOW_ELEMENTARY_UNARY_ATTRS_H

#include "core.h"
#include "op-attrs/op.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct ElementScalarUnaryAttrs {
  Op op;
  req<float> scalar;
};
FF_VISITABLE_STRUCT(ElementScalarUnaryAttrs, op, scalar);
FF_VISIT_FMTABLE(ElementScalarUnaryAttrs);

CHECK_VALID_OP_ATTR(ElementScalarUnaryAttrs);

struct ElementUnaryAttrs {
  req<Op> op;
};
FF_VISITABLE_STRUCT(ElementUnaryAttrs, op);
FF_VISIT_FMTABLE(ElementUnaryAttrs);

CHECK_VALID_OP_ATTR(ElementUnaryAttrs);

} // namespace FlexFlow

#endif
