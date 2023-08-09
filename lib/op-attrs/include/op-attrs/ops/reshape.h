#ifndef _FLEXFLOW_RESHAPE_ATTRS_H
#define _FLEXFLOW_RESHAPE_ATTRS_H

#include "op-attrs/tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct ReshapeAttrs : public use_visitable_cmp<ReshapeAttrs> {
public:
  ReshapeAttrs() = delete;
  explicit ReshapeAttrs(TensorShape const &shape);

public:
  TensorShape shape;
};

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::ReshapeAttrs, shape);
MAKE_VISIT_HASHABLE(::FlexFlow::ReshapeAttrs);

#endif
