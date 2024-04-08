#ifndef _FLEXFLOW_ELEMENTARY_UNARY_ATTRS_H
#define _FLEXFLOW_ELEMENTARY_UNARY_ATTRS_H

#include "core.h"
#include "op-attrs/ops/element_scalar_unary_attrs.h"
#include "op-attrs/ops/element_unary_attrs.h"
#include "op-attrs/parallel_tensor_shape.h"

namespace FlexFlow {

CHECK_VALID_OP_ATTR(ElementUnaryAttrs);
CHECK_VALID_OP_ATTR(ElementScalarUnaryAttrs);

} // namespace FlexFlow

#endif
