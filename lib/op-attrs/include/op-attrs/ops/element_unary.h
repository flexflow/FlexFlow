#ifndef _FLEXFLOW_ELEMENTARY_UNARY_ATTRS_H
#define _FLEXFLOW_ELEMENTARY_UNARY_ATTRS_H

#include "core.h"
#include "op-attrs/op.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct ElementUnaryAttrs {
  req<Op> op_type;
};
FF_VISITABLE_STRUCT(ElementUnaryAttrs, op_type);
CHECK_VALID_OP_ATTR(ElementUnaryAttrs);

struct ElementScalarUnaryAttrs {
    Op op_type;
  req<float> scalar;
};
FF_VISITABLE_STRUCT(ElementScalarUnaryAttrs, op_type, scalar);
CHECK_VALID_OP_ATTR(ElementScalarUnaryAttrs);

} // namespace FlexFlow

#endif
