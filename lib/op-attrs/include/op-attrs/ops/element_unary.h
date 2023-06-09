#ifndef _FLEXFLOW_ELEMENTARY_UNARY_ATTRS_H
#define _FLEXFLOW_ELEMENTARY_UNARY_ATTRS_H

#include "core.h"
#include "op-attrs/op.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct ElementScalarUnaryAttrs
    : public use_visitable_cmp<ElementScalarUnaryAttrs> {
public:
  ElementScalarUnaryAttrs(Op, float);

public:
  Op op;
  /* bool inplace; */
  float scalar;
};

struct ElementUnaryAttrs : public use_visitable_cmp<ElementUnaryAttrs> {
public:
  ElementUnaryAttrs(OperatorType);

public:
  Op op;
};

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::ElementScalarUnaryAttrs, op, scalar);
MAKE_VISIT_HASHABLE(::FlexFlow::ElementScalarUnaryAttrs);

VISITABLE_STRUCT(::FlexFlow::ElementUnaryAttrs, op);
MAKE_VISIT_HASHABLE(::FlexFlow::ElementUnaryAttrs);

#endif
